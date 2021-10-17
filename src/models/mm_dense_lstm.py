from models.mm_dense_rnn import MMDenseRNN

FULL = 'full'
EPS = 1e-12

class MMDenseLSTM(MMDenseRNN):
    def __init__(
        self, 
        in_channels, num_features,
        growth_rate,
        kernel_size,
        bands=['low','middle'], sections=[512,513],
        scale=(2,2),
        dilated=False, norm=True, nonlinear='relu',
        depth=None,
        growth_rate_final=None,
        kernel_size_final=None,
        dilated_final=False,
        norm_final=True, nonlinear_final='relu',
        depth_final=None,
        rnn_position='parallel',
        eps=EPS,
        **kwargs
    ):
        super().__init__(
            in_channels, num_features, growth_rate, kernel_size,
            bands=bands, sections=sections,
            scale=scale, dilated=dilated, norm=norm, nonlinear=nonlinear,
            depth=depth,
            growth_rate_final=growth_rate_final,
            kernel_size_final=kernel_size_final,
            dilated_final=dilated_final,
            norm_final=norm_final, nonlinear_final=nonlinear_final,
            depth_final=depth_final,
            rnn_type='lstm', rnn_position=rnn_position,
            eps=eps,
            **kwargs
        )