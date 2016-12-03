    def loss(sf):
        L = 0
        # For each example
        for i in np.arange(len(sf.label)):
            # For each timestep
            output = np.zeros([len(sf.label[i]),256])
            target = np.zeros(len(sf.label[i]))
            for t in range(len(sf.label[i])):
                asciiChar_input = ord(sf.allData[i][t])
                sf.calc_y_hidden_layer(i, t, calc_type=0)
                asciiChar_predict = ord(sf.calc_y_softmax_output_layer(0, t, calc_type=0))
                asciiChar_target = ord(sf.label[i][t])
                target[t] = asciiChar_target
                output[t,asciiChar_predict] = 1

            correct_word_predictions = output[:, target.astype(int)]
            
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L
