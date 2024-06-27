class predictModel:
    def __init__(self):
        self._mae = ''
        self._mape = ''
        self._mse = ''
        self._r2 = ''
        self._tuningimprovement = ''
        self._X = '' 
        self._y = ''
        self._X_train = ''
        self._X_test = ''
        self._y_train = ''
        self.y_test = ''
        self._specified_categories = []
        self._mae_values = []
        self._mse_values = []
        self._r2_values = []
        self._mape_values = []
        self._tuning_improvement_value = []
    
    def get_mae(self):
        return self._mae
    
    def set_mae(self, value):
        self._mae = value
    
    def get_mape(self):
        return self._mape
    
    def set_mape(self, value):
        self._mape = value
    
    def get_mse(self):
        return self._mse
    
    def set_mse(self, value):
        self._mse = value
    
    def get_r2(self):
        return self._r2
    
    def set_r2(self, value):
        self._r2 = value

    def get_tuningimprovement(self):
        return self._tuningimprovement
    
    def set_tuningimprovement(self, value):
        self._tuningimprovement = value

    def get_X(self):
        return self._X
    
    def set_X(self, value):
        self._X = value

    def get_y(self):
        return self._y
    
    def set_y(self, value):
        self._y = value
        
    def get_X_train(self):
        return self._X_train
    
    def set_X_train(self, value):
        self._X_train = value

    def get_X_test(self):
        return self._X_test
    
    def set_X_test(self, value):
        self._X_test = value

    def get_y_train(self):
        return self._y_train
    
    def set_y_train(self, value):
        self._y_train = value  

    def get_y_test(self):
        return self._y_test
    
    def set_y_test(self, value):
        self._y_test = value  

    @property
    def specified_categories(self):
        return self._specified_categories

    @specified_categories.setter
    def specified_categories(self, value):
        if not isinstance(value, list):
            raise ValueError("Specified categories must be a list")
        self._specified_categories = value

    @property
    def mae_values(self):
        return self._mae_values

    @mae_values.setter
    def mae_values(self, value):
        if not isinstance(value, list):
            raise ValueError("MAE values must be a list")
        self._mae_values = value

    @property
    def mse_values(self):
        return self._mse_values

    @mse_values.setter
    def mse_values(self, value):
        if not isinstance(value, list):
            raise ValueError("MSE values must be a list")
        self._mse_values = value

    @property
    def r2_values(self):
        return self._r2_values

    @r2_values.setter
    def r2_values(self, value):
        if not isinstance(value, list):
            raise ValueError("R2 values must be a list")
        self._r2_values = value

    @property
    def mape_values(self):
        return self._mape_values

    @mape_values.setter
    def mape_values(self, value):
        if not isinstance(value, list):
            raise ValueError("MAPE values must be a list")
        self._mape_values = value

    @property
    def tuning_improvement_value(self):
        return self._tuning_improvement_value

    @tuning_improvement_value.setter
    def tuning_improvement_value(self, value):
        if not isinstance(value, list):
            raise ValueError("Tuning improvement must be a list")
        self._tuning_improvement_value = value