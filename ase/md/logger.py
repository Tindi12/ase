from ase.logger import Logger as RealLogger
from ase.utils import deprecated


class MDLogger(RealLogger):
    @deprecated('`MDLogger` have been changed to `Logger` in ase.logger')
    def __init__(self, *args, **kwargs):
        """
        .. deprecated:: 3.23.0
            Please import :class:`~ase.logger.Logger` from :mod:`ase.logger`
        """
        super().__init__(*args, **kwargs)
