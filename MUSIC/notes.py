labels = np.array([50])
    M = 100
    Nq = 5
    SNR = 0
    snapshot = 500
    A = Matrix_class(M, labels).matrix()
    B = Matrix_class(M, np.array([0])).matrix()
    my_vec = observ(SNR, snapshot, A)
    R = np.cov(quantize(my_vec, 0))
    # eigvals, eigvecs = np.linalg.eig(R)
    # sorted_indices = np.argsort(eigvals.real)[::-1]  # Sort eigenvalues in descending order
    # eigvecs_sorted = eigvecs[:, sorted_indices]
    # Es = eigvecs_sorted[:, :2]
    # En = eigvecs_sorted[:, 2:]
    #
    # my_vec_mixed = quantize(my_vec, Nq)
    # R_mixed = np.cov(my_vec_mixed)
    # eigvals_mixed, eigvecs_mixed = np.linalg.eig(R_mixed)
    # sorted_indices_mixed = np.argsort(eigvals_mixed.real)[::-1]  # Sort eigenvalues in descending order
    # eigvecs_sorted_mixed = eigvecs_mixed[:, sorted_indices_mixed]
    # Es_mixed = eigvecs_sorted_mixed[:, :2]
    # En_mixed = eigvecs_sorted_mixed[:, 2:]
    #
    # my_vec_qun = quantize(my_vec, M)
    # R_qun = np.cov(my_vec_qun)
    # eigvals_qun, eigvecs_qun = np.linalg.eig(R_qun)
    # sorted_indices_qun = np.argsort(eigvals_qun.real)[::-1]  # Sort eigenvalues in descending order
    # eigvecs_sorted_qun = eigvecs_qun[:, sorted_indices_qun]
    # Es_qun = eigvecs_sorted_qun[:, :2]
    # En_qun = eigvecs_sorted_qun[:, 2:]

    # print("ESPRIT mixed:",LA.norm(Es-Es_mixed, "fro"))
    # print("ESPRIT qun:",LA.norm(Es-Es_qun, "fro"))
    # print("========================================")
    # print("MUSIC mixed:",LA.norm(En-En_mixed, "fro"))
    # print("MUSIC qun:",LA.norm(En-En_qun, "fro"))