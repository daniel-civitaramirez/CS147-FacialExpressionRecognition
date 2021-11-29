class ConvulationParameters:
    def __init__(self, features, kernel, strides, padding, activation=None):
        self.features = features
        self.kernel = kernel
        self.strides = strides
        self.padding = padding

    def GetString(self):
        return f"features - {self.features}, kernel - {self.kernel}, strides - {self.strides}, padding - {self.padding}"


class MaxPoolParameters:
    def __init__(self, pool, stride):
        self.pool = pool
        self.stride = stride

    def GetString(self):
        return f"pool - {self.pool}, stride - {self.stride}"
