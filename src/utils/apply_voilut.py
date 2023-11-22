import numpy as np

class DicomsdlMetadata:
    def __init__(self, ds):
        self.window_widths = ds.WindowWidth
        self.window_centers = ds.WindowCenter
        if self.window_widths is None or self.window_centers is None:
            self.window_widths = []
            self.window_centers = []
        else:
            try:
                if not isinstance(self.window_widths, list):
                    self.window_widths = [self.window_widths]
                self.window_widths = [float(e) for e in self.window_widths]
                if not isinstance(self.window_centers, list):
                    self.window_centers = [self.window_centers]
                self.window_centers = [float(e) for e in self.window_centers]
            except:
                self.window_widths = []
                self.window_centers = []

        self.voilut_func = ds.VOILUTFunction
        if self.voilut_func is None:
            self.voilut_func = 'LINEAR'
        else:
            self.voilut_func = str(self.voilut_func).upper()
        self.invert = (ds.PhotometricInterpretation == 'MONOCHROME1')
        assert len(self.window_widths) == len(self.window_centers)

class ApplyWindowing:
    def apply_windowing_np_v2(arr, window_width=None, window_center=None, voi_func='LINEAR', y_min=0, y_max=255):
        assert window_width > 0
        y_range = y_max - y_min
        arr = arr.astype(np.float32)

        if voi_func == 'LINEAR' or voi_func == 'LINEAR_EXACT':
            if voi_func == 'LINEAR':
                if window_width < 1:
                    raise ValueError(
                        "The (0028,1051) Window Width must be greater than or "
                        "equal to 1 for a 'LINEAR' windowing operation")
                window_center -= 0.5
                window_width -= 1

            s = y_range / window_width
            b = (-window_center / window_width + 0.5) * y_range + y_min
            arr = arr * s + b
            arr = np.clip(arr, y_min, y_max)

        elif voi_func == 'SIGMOID':
            s = -4 / window_width
            arr = y_range / (1 + np.exp((arr - window_center) * s)) + y_min
        else:
            raise ValueError(
                f"Unsupported (0028,1056) VOI LUT Function value '{voi_func}'")
        return arr
    
    def min_max_scale(img):
        maxv = img.max()
        minv = img.min()
        if maxv > minv:
            return (img - minv) / (maxv - minv)
        else:
            return img - minv