"""[codex] Durable external reading journal before opening any model artifact."""
from contextlib import contextmanager
import fcntl,hashlib,json,os,re,uuid
from pathlib import Path
from codex.retrieval_matrix_analysis import contracts as C

PREFIX=b'[codex] '
FIELDS={'time','event','actor','token','plan_sha256','analysis_commit','job','producer_plan_sha256','paths','previous','error_type','hash_errors'}
EVENTS={'READ_STARTED','READ_COMPLETED','READ_FAILED'}
ANCHOR_ROOT=C.ROOT/'dl/retrieval-matrix-analysis/reading-anchor'

def _parse(body,*,expected_plan=None,expected_commit=None):
    if body.count(C.MARKER)!=1:raise ValueError('One immutable reading-log boundary is required')
    _,tail=body.split(C.MARKER);rows=[];previous=None;opened={};seen=set()
    for line in tail.splitlines():
        if not line.startswith(PREFIX):raise ValueError('Only machine reading records may follow the frozen preregistration')
        row=json.loads(line[len(PREFIX):])
        if (set(row)!=FIELDS or row['event'] not in EVENTS or row['previous']!=previous or not row['actor']
                or (expected_plan is not None and row['plan_sha256']!=expected_plan)
                or (expected_commit is not None and row['analysis_commit']!=expected_commit)):
            raise ValueError('Reading record schema, identity or append chain differs')
        if row['event']=='READ_STARTED':
            if row['token'] in seen:raise ValueError('Reading token reused')
            if any(h is not None for h in row['paths'].values()):raise ValueError('Read start must precede reading artifact hashes')
            opened[row['token']]=row;seen.add(row['token'])
        else:
            if row['token'] not in opened:raise ValueError('Reading completion lacks an earlier durable start')
            start=opened.pop(row['token'])
            if any(row[k]!=start[k] for k in ('actor','job','producer_plan_sha256')) or set(row['paths'])!=set(start['paths']):
                raise ValueError('Reading completion belongs to a different actor, job or path scope')
        if not isinstance(row['paths'],dict):raise ValueError('Reading record requires explicit paths and post-read hashes')
        if not isinstance(row['hash_errors'],dict) or not set(row['hash_errors'])<=set(row['paths']):raise ValueError('Hash errors must name declared read paths')
        if row['event']!='READ_FAILED' and (row['hash_errors'] or row['error_type'] is not None):raise ValueError('Only failed reading records may carry errors')
        for p,h in row['paths'].items():
            C.safe(p)
            if h is not None and (not isinstance(h,str) or re.fullmatch('[0-9a-f]{64}',h) is None):raise ValueError('Malformed read hash')
        rows.append(row);previous=C.digest(row)
    return rows

def _encode(value,prefix=b''):
    return prefix+json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'

def _point(body,rows):
    return {'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),'records':len(rows),
        'head':C.digest(rows[-1]) if rows else None}

def _fsync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def _ensure_directory(path):
    path=C.safe(path)
    if not path.exists():
        _ensure_directory(path.parent)
        try:path.mkdir()
        except FileExistsError:pass
        _fsync_directory(path.parent)

@contextmanager
def _anchor_lock(path,anchor_root=None):
    root=C.safe(anchor_root if anchor_root is not None else ANCHOR_ROOT)
    directory=root/C.digest({'prereg_path':str(C.safe(path))})
    _ensure_directory(directory)
    with (directory/'writer.lock').open('a+b') as lock:
        _fsync_directory(directory)
        fcntl.flock(lock,fcntl.LOCK_EX)
        yield directory/'journal.jsonl'

def _append_anchor(path,value,*,create=False):
    # No replace/truncate/reset path exists. A missing established journal is not recreated.
    flags=os.O_WRONLY|os.O_APPEND|(os.O_CREAT|os.O_EXCL if create else 0)
    fd=os.open(path,flags,0o600)
    with os.fdopen(fd,'ab') as stream:
        stream.write(_encode(value));stream.flush();os.fsync(stream.fileno())
    if create:_fsync_directory(path.parent)

def _append_prereg(path,payload):
    # The source already exists. Missing preregistrations must never be created here.
    fd=os.open(path,os.O_WRONLY|os.O_APPEND)
    with os.fdopen(fd,'ab') as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())

def _checked_anchor(path,journal,*,expected_plan=None,expected_commit=None):
    body=Path(path).read_bytes();rows=_parse(body,expected_plan=expected_plan,expected_commit=expected_commit)
    header,tail=C.prereg_body(path)
    if not journal.exists():
        if tail:raise ValueError('A nonempty reading log without its external anchor cannot be adopted')
        return rows,None
    raw=journal.read_bytes()
    if not raw or not raw.endswith(b'\n'):raise ValueError('Incomplete reading-anchor journal; reviewed recovery required')
    try:events=[json.loads(line) for line in raw.splitlines()]
    except (ValueError,UnicodeError) as error:raise ValueError('Corrupt reading-anchor journal; reviewed recovery required') from error
    first=events[0]
    if (set(first)!={'schema','owner','kind','identity','checkpoint','previous'} or first['schema']!=1
            or first['owner']!='[codex]' or first['kind']!='GENESIS' or first['previous'] is not None):
        raise ValueError('External reading-anchor genesis differs')
    identity=first['identity']
    if (set(identity)!={'prereg_path','body_sha256','plan_sha256','analysis_commit'}
            or identity['prereg_path']!=str(C.safe(path)) or identity['body_sha256']!=hashlib.sha256(header).hexdigest()
            or (expected_plan is not None and identity['plan_sha256']!=expected_plan)
            or (expected_commit is not None and identity['analysis_commit']!=expected_commit)):
        raise ValueError('External reading anchor belongs to a different preregistration, plan, body or commit')
    expected_body=header;expected_rows=[];point=_point(expected_body,expected_rows);pending=None
    if first['checkpoint']!=point:raise ValueError('External reading-anchor genesis checkpoint differs')
    previous=C.digest(first)
    for event in events[1:]:
        if event.get('previous')!=previous:raise ValueError('External reading-anchor hash chain differs')
        if event.get('kind')=='PREPARE' and pending is None:
            if set(event)!={'kind','previous','before','after','event'} or event['before']!=point:
                raise ValueError('External reading-anchor expected previous head/length differs')
            next_body=expected_body+_encode(event['event'],PREFIX)
            next_rows=_parse(next_body,expected_plan=identity['plan_sha256'],expected_commit=identity['analysis_commit'])
            if event['after']!=_point(next_body,next_rows):raise ValueError('External pending event/head/length differs')
            pending=(C.digest(event),next_body,next_rows,event['after'])
        elif event.get('kind')=='COMMIT' and pending is not None:
            if set(event)!={'kind','previous','prepared_sha256','checkpoint'} or event['prepared_sha256']!=pending[0] or event['checkpoint']!=pending[3]:
                raise ValueError('External reading-anchor commit differs from its prepared event')
            _,expected_body,expected_rows,point=pending;pending=None
        else:raise ValueError('External reading-anchor transaction order differs')
        previous=C.digest(event)
    if pending is not None:raise ValueError('Incomplete reading-anchor transaction; reviewed recovery required')
    if body!=expected_body or _point(body,rows)!=point:
        raise ValueError('Reading log was truncated, rewritten or rolled back relative to its durable external anchor')
    return rows,{'identity':identity,'checkpoint':point,'head':previous}

def records(path,*,expected_plan=None,expected_commit=None,anchor_root=None):
    with _anchor_lock(path,anchor_root) as journal:
        rows,_=_checked_anchor(path,journal,expected_plan=expected_plan,expected_commit=expected_commit)
        return rows

def append(path,*,plan_sha,commit,job,actor,event,token,paths,error_type=None,hash_errors=None,anchor_root=None):
    with _anchor_lock(path,anchor_root) as journal:
        prior,anchor=_checked_anchor(path,journal,expected_plan=plan_sha,expected_commit=commit)
        row={'time':C.now(),'event':event,'actor':actor,'token':token,'plan_sha256':plan_sha,
            'analysis_commit':commit,'job':job['id'],'producer_plan_sha256':job['producer_plan']['sha256'],
            'paths':paths,'previous':C.digest(prior[-1]) if prior else None,'error_type':error_type,'hash_errors':hash_errors or {}}
        old_body=Path(path).read_bytes();payload=_encode(row,PREFIX);next_body=old_body+payload
        next_rows=_parse(next_body,expected_plan=plan_sha,expected_commit=commit)
        if anchor is None:
            header,_=C.prereg_body(path)
            genesis={'schema':1,'owner':'[codex]','kind':'GENESIS','identity':{'prereg_path':str(C.safe(path)),
                'body_sha256':hashlib.sha256(header).hexdigest(),'plan_sha256':plan_sha,'analysis_commit':commit},
                'checkpoint':_point(old_body,prior),'previous':None}
            _append_anchor(journal,genesis,create=True)
            anchor={'identity':genesis['identity'],'checkpoint':genesis['checkpoint'],'head':C.digest(genesis)}
        prepared={'kind':'PREPARE','previous':anchor['head'],'before':anchor['checkpoint'],
            'after':_point(next_body,next_rows),'event':row}
        # PREPARE is durable before touching the preregistration or permitting artifact reads.
        _append_anchor(journal,prepared)
        _append_prereg(path,payload)
        _append_anchor(journal,{'kind':'COMMIT','previous':C.digest(prepared),'prepared_sha256':C.digest(prepared),'checkpoint':prepared['after']})
        _checked_anchor(path,journal,expected_plan=plan_sha,expected_commit=commit)
    return C.digest(row)

def require_completed(path,started,*,plan_sha,commit,anchor_root=None):
    rows=records(path,expected_plan=plan_sha,expected_commit=commit,anchor_root=anchor_root)
    starts=[r for r in rows if C.digest(r)==started and r['event']=='READ_STARTED']
    if len(starts)!=1:raise ValueError('Mechanical stage lacks its registered reading start')
    end=[r for r in rows if r['token']==starts[0]['token'] and r['event']!='READ_STARTED']
    if len(end)!=1 or end[0]['event']!='READ_COMPLETED':raise ValueError('Mechanical stage needs a successfully completed reading record')
    return end[0]

class ReadSession:
    def __init__(self,plan_path,commit,job,actor,paths,*,anchor_root=None):
        self.args={'path':C.HERE/'PREREG.md','plan_sha':C.sha(plan_path),'commit':commit,'job':job,'actor':actor,'token':str(uuid.uuid4()),'anchor_root':anchor_root}
        self.paths=[str(C.safe(p)) for p in paths]
    def __enter__(self):
        self.start=append(**self.args,event='READ_STARTED',paths={p:None for p in self.paths});return self
    def __exit__(self,kind,error,tb):
        hashes={};errors={};hash_exception=None
        for p in self.paths:
            try:
                if not Path(p).is_file():
                    if kind is None:raise FileNotFoundError('Declared successful read artifact is missing: '+p)
                    hashes[p]=None
                else:hashes[p]=C.sha(p)
            except OSError as failure:
                hashes[p]=None;errors[p]=type(failure).__name__
                if hash_exception is None:hash_exception=failure
        failed=kind is not None or hash_exception is not None
        self.end=append(**self.args,event='READ_FAILED' if failed else 'READ_COMPLETED',paths=hashes,
            error_type=kind.__name__ if kind else (type(hash_exception).__name__ if hash_exception else None),hash_errors=errors)
        if hash_exception is not None and kind is None:raise hash_exception
        return False
