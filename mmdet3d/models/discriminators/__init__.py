from .disc import (FCDiscriminator, FCDiscriminatorNew, FCDiscriminatorCE,
                   Conv2dDiscriminator, Conv2dDiscriminator01, DetDiscriminator,
                   ConvDiscriminator1x1)
from .consistency_disc import ConsistencyDisc
__all__ = ['FCDiscriminator', 'FCDiscriminatorCE', 'FCDiscriminatorNew',
           'Conv2dDiscriminator', 'Conv2dDiscriminator01', 'DetDiscriminator',
           'ConvDiscriminator1x1', 'ConsistencyDisc']
