from toolsbench.data.base import DeepinvData


class HighResColorImagingData(DeepinvData):

    @property
    def image_name(self) -> str:
        return "butterfly.png"


class Tomography2D(DeepinvData):

    @property
    def image_name(self) -> str:
        return "SheppLogan.png"
