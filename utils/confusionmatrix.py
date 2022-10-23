class ConfusionMatrix:
    """Confusion Matrix
    """
    def __init__(self, all_output):
        self.all_output = all_output
        self.actual = []
        self.predicted = []
        self.column = [0] * 4
        self.accuracy = 0.0
        self.misclass = 0.0
    
    def add_data(self, actual, predicted):
        """add actual, predicted data into confusion matrix

        Args:
            actual (number): actual output
            predicted (number): predicted output
        """
        self.actual.append(actual)
        self.predicted.append(predicted)
        
    def calc_column(self):
        """calculating confusion matrix
        """
        for i in range(0, len(self.actual), 1):
            # actual 0, predict 0
            if (self.all_output[0] == self.actual[i]) and (self.all_output[0] == self.predicted[i]) and (self.actual[i] == self.predicted[i]):
                self.column[0] += 1
            # actual 1, predict 1
            elif (self.all_output[1] == self.actual[i]) and (self.all_output[1] == self.predicted[i]) and (self.actual[i] == self.predicted[i]):
                self.column[3] += 1
            # actual 0, predict 1
            elif (self.all_output[0] == self.actual[i]) and (self.actual[i] != self.predicted[i]):
                self.column[1] += 1
            # actual 1, predict 0
            elif (self.all_output[1] == self.actual[i]) and (self.actual[i] != self.predicted[i]):
                self.column[2] += 1
            self.accuracy = float((self.column[0] + self.column[3]) / len(self.actual))
            self.misclass = float((self.column[1] + self.column[2]) / len(self.actual))
    
    def print(self):
        """printing result
        """
        print('                             Predicted')
        print('                        +--------+--------+')
        print('                        |',self.all_output[0],'|', self.all_output[1],'|')
        print('               +--------+--------+--------+')
        print('               |', self.all_output[0], '|  ', self.column[0], '      ', self.column[1], '      ')
        print('      Actual   +--------+')
        print('               |', self.all_output[1], '|  ', self.column[2], '      ', self.column[3], '      ')
        print('               +--------+')
        
    def get_accuracy(self):
        """get accuracy value

        Returns:
            float: accuracy value
        """
        return self.accuracy
    
    def get_misclass(self):
        """get misclass value

        Returns:
            float: misclass value
        """
        return self.misclass
    
    def clear(self):
        """clear data in confusion matrix
        """
        self.actual = []
        self.predicted = []
        self.column = [0] * 4
        self.accuracy = 0.0
        self.misclass = 0.0
        
    def get(self):
        """get confusion matrix in list form

        Returns:
            list: confusion matrix in list form
        """
        form_array = [[self.column[0],self.column[1]],[self.column[2],self.column[3]]]
        return form_array