# [codex] Paired CLIC runtime repair

保留三個54f544c原paired模型／資料／FP32 batch1與作者strict公式，只另補einops及signature repr地址穩定性。新private producer／paired qualification不修改既有frozen modules；CPU直接測flashpadding，不以CUDAhidden lazyimports代替驗證。

central_task.sh FULL_REF PLAN MODEL OUT供root新批次使用，輸出限dl/retrieval-matrix-paired-runtime-fix/runs/的新子目錄。新plan須另review與freeze；新qualification scope拒絕舊P0與Google資格。沒有GPU／排程／實測資格。本PREREG完整列明來源、signature政策與等同性鏈。
