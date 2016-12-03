    def adagrad_weight_update(sf):
        grad_prod = np.dot(sf.gradDescHidden, sf.gradDescHidden.T)
        diag = grad.diagonal()
        adagrad = 1/np.sqrt(diag)
        for col in range(sf.gradDescHidden.shape[1]):
            sf.gradDescHidden[:,col] = sf.gradDescHidden[:,col] * adagrad
        sf.weightsIJ += np.dot(sf.learningRate, sf.gradDescHidden)

        grad_prod = np.dot(sf.gradDescOutput, sf.gradDescOutput.T)
        diag = grad.diagonal()
        adagrad = 1/np.sqrt(diag)
        for col in range(sf.gradDescOutput.shape[1]):
            sf.gradDescOutput[:,col] = sf.gradDescOutput[:,col] * adagrad
        sf.weightsJK += np.dot( sf.learningRate, sf.gradDescOutput )

        grad_prod = np.dot(sf.gradDescHH, sf.gradDescHH.T)
        diag = grad.diagonal()
        adagrad = 1/np.sqrt(diag)
        for col in range(sf.gradDescHH.shape[1]):
            sf.gradDescHH[:,col] = sf.gradDescHH[:,col] * adagrad
        sf.weightsHH += np.dot( sf.learningRate, sf.gradDescHH )
