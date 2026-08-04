from typing import Any, Optional


class PipelineError(Exception):
    def __init__(self, message: str, details: Optional[Any] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class DataError(PipelineError):
    pass


class ModelError(PipelineError):
    pass


class PreprocessingError(PipelineError):
    pass


class ValidationError(PipelineError):
    pass


class FileOperationError(PipelineError):
    pass
