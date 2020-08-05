import math
from functools import partial
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
# import settings.settings as settings


class ChannelAlignment(nn.Module):
    """
    Given two input tensors, adjusts the number of channels of the shallower one to match the deeper one. Required for
    element-wise addition.
    """

    def __init__(self, in_channels: list):
        super(ChannelAlignment, self).__init__()
        self.in_channels = in_channels
        self.out_ch = max(self.in_channels)

        self.module_dict = {}
        for i, in_ch in enumerate(self.in_channels):
            if in_ch < self.out_ch:
                self.module_dict[str(i)] = nn.Conv2d(in_ch, self.out_ch, (1, 1), bias=False)

        self.module_dict = nn.ModuleDict(self.module_dict)

    def forward(self, inputs: list):
        # shapes = [ag.Variable(x.data, requires_grad=False).shape for x in inputs]
        # assert all(s[1] == self.in_channels[i] for i, s in enumerate(shapes)), \
        #     "Inconsistency between tensors and declared sizes"
        out = []
        for i, x in enumerate(inputs):
            if str(i) in self.module_dict:
                out.append(self.module_dict[str(i)](x))
            else:
                out.append(x)

        # assert all(x.shape[1] == self.out_ch for x in out), \
        #     "Something went wrong, all tensors do not have the same number of channels."
        return out


class Aggregation(nn.Module):
    """
    Base class for aggregation functions.
    If input and output have different sizes,
    use largest in each dimension and zero-pad (spatial dimensions), or convolve with a 1x1 filter
    (number of channels)
    """

    def __init__(self):
        """
        :param agg_params: Parameters for the aggregation operation. Contains:
            - 'pad_or_interpolate': Whether to use padding (0) or interpolation (1) to bring both inputs to the spatial
              dimensions of the largest input (0: pad, 1: interpolate)
            - 'interpolate_mode': Mode used for the feature map interpolations, either 0 (nearest) or 1 (bilinear)
        """
        super(Aggregation, self).__init__()

    def forward(self, inputs: list):
        raise NotImplementedError("Abstract method")

    def align_sizes_pad(self, inputs: list, mode = 'replicate'):  # x1: torch.Tensor, x2: torch.Tensor):
        """
        If inputs have different spatial dimensions (W, H), add padding to the smaller ones. If the total amount of
        padding to add along an axis is not even, the padding is greater at the right-hand side and bottom of each
        channel.
        :param inputs: A list of torch.Tensors with spatial dimensions (N, C, H_i, W_i)
        :param mode: Type of padding to use: 'constant' (zeros), 'reflect', 'replicate' (default)
        :return: A list of torch.Tensors with identical spatial dimensions (N, sum(C_i), H_max, W_max)
        """
        assert mode in ['constant', 'replicate', 'reflect'], "Error: Unknown padding mode {}".format(mode)

        heights = [x.detach().shape[2] for x in inputs]
        widths = [x.detach().shape[3] for x in inputs]
        h_max, w_max = max(heights), max(widths)

        # check that we can apply padding (the padding size needs to be smaller than the input size when using reflect)
        h_crit = all([int(math.ceil(max(h_max - heights[i], 0) / 2.)) < heights[i] for i in range(len(inputs))])
        w_crit = all([int(math.ceil(max(w_max - widths[i], 0) / 2.)) < widths[i] for i in range(len(inputs))])
        if mode == 'reflect' and not (h_crit and w_crit): # if not, we have to interpolate
            warnings.warn("Tensor size was smaller than required padding; using Interpolate instead of Pad."
                          "You might want to consider using 'constant' or 'replicate' padding.",
                          UserWarning)
            return self.align_sizes_interpolate(inputs, mode='nearest')

        # Otherwise:
        for i in range(len(inputs)):
            h_diff = max(h_max - heights[i], 0)
            w_diff = max(w_max - widths[i], 0)

            if h_diff > 0:
                inputs[i] = F.pad(inputs[i], [0, 0, int(math.floor(h_diff / 2.)), int(math.ceil(h_diff / 2.))],
                                  mode=mode)
            if w_diff > 0:
                inputs[i] = F.pad(inputs[i], [int(math.floor(w_diff / 2.)), int(math.ceil(w_diff / 2.)), 0, 0],
                                  mode=mode)

        assert all(x.detach().shape[2] == h_max for x in inputs) and all(x.detach().shape[3] == w_max for x in inputs), \
            "Something went wrong, all outputs do not have the same size."

        return inputs

    @staticmethod
    def align_sizes_interpolate(inputs: list, mode: str = 'nearest'):
        """
        If inputs have different spatial dimensions (W, H), interpolate the smaller ones.
        :param inputs: A list of torch.Tensors with spatial dimensions (N, C, H_i, W_i)
        :param mode: Type of interpolation to use: 'nearest' (default) or 'bilinear'
        :return: A list of torch.Tensors with identical spatial dimensions (N, sum(C_i), H_max, W_max)
        """
        assert mode in ['nearest', 'bilinear'], "Error: Unknown interpolation mode {}".format(mode)
    
        heights = [x.detach().size(2) for x in inputs]
        widths = [x.detach().size(3) for x in inputs]
        h_max, w_max = max(heights), max(widths)

        for i in range(len(inputs)):
            h_diff = max(h_max - inputs[i].size(2), 0)
            w_diff = max(w_max - inputs[i].size(3), 0)

            if h_diff > 0:
                inputs[i] = F.interpolate(inputs[i], (h_max, w_max), mode=mode, align_corners=True)

            if w_diff > 0:
                inputs[i] = F.interpolate(inputs[i], (h_max, w_max), mode=mode, align_corners=True)

        assert all(x.detach().shape[2] == h_max for x in inputs) and all(x.shape[3] == w_max for x in inputs), \
            "Something went wrong, all outputs do not have the same size."

        return inputs


class Addition(Aggregation):
    """
    Add two input tensors, return a single output tensor of same dimensions. If input and output have different sizes,
    use largest in each dimension and zero-pad or interpolate (spatial dimensions), or convolve with a 1x1 filter
    (number of channels)
    """
    def __init__(self, in_channels: list, pad_or_interpolate: str = 'pad', pad_mode: str = 'replicate', 
        interpolate_mode: str = 'nearest'):

        assert pad_or_interpolate in ['pad', 'interpolate', 'inter'], \
        "Error: Unknown value for `pad_or_interpolate` {}".format(pad_or_interpolate)

        super(Addition, self).__init__()
        self.ch_align = ChannelAlignment(in_channels)

        if pad_or_interpolate == 'pad':
            self.sz_align = partial(self.align_sizes_pad, mode=pad_mode)
        else: 
            self.sz_align = partial(self.align_sizes_interpolate, mode=interpolate_mode)

    def forward(self, inputs: list):
        """
        Performs element-wise sum of inputs. If they have different dimensions, they are first adjusted to
        common dimensions by 1/ padding or interpolation (h and w axes) and/or 2/ 1x1 convolution.
        :param inputs: List of torch input tensors of dimensions (N, C_i, H_i, W_i)
        :return: A single torch Tensor of dimensions (N, max(C_i), max(H_i), max(W_i)), containing the element-
            wise sum of the input tensors (or their size-adjusted variants)
        """
        inputs = self.sz_align(inputs)  # Perform size alignment
        inputs = self.ch_align(inputs)  # Perform channel alignment
        stacked = torch.stack(inputs, dim=4)  # stack inputs along an extra axis (will be removed when summing up)
        return torch.sum(stacked, 4, keepdim=True).squeeze(4)

class Concatenation(Aggregation):
    """
    Add two input tensors, return a single output tensor of same dimensions. If input and output have different sizes,
    use largest in each dimension and zero-pad or interpolate (spatial dimensions), or convolve with a 1x1 filter
    (number of channels)
    """

    def __init__(self, pad_or_interpolate: str = 'pad', pad_mode: str = 'replicate', interpolate_mode: str = 'nearest'):
        """
        :param pad_or_interpolate: 'pad' or 'interpolate'. When inputs are of unequal spatial dimensions, which method
            to use to align their sizes (H, W).
        :param pad_mode: 'constant' (zeros), 'replicate' or 'reflect'. If using padding, which method of padding to use.
            Ignored if pad_or_interpolate == 'interpolate'
        :param interpolate_mode: If using interpolation, which method of interpolation to use. Ignored if
            pad_or_interpolate == 'pad'.
        """
        super(Concatenation, self).__init__()
        assert pad_or_interpolate in ['pad', 'inter', 'interpolate']
        if pad_or_interpolate == 'pad':
            self.sz_align = partial(self.align_sizes_pad, mode=pad_mode)
        else: 
            self.sz_align = partial(self.align_sizes_interpolate, mode=interpolate_mode)

    def forward(self, inputs: list):
        """
        Performs concatenation of inputs x1 and x2 along the channel axis. If they have different heights or widths,
        they are first adjusted to common dimensions by padding or interpolation.
        :param inputs: List of torch input tensor of dimensions (N, c_i, h_i, w_i)
        :return: A single torch Tensor of dimensions (N, sum(c_i), max(h_i), max(w_i)), containing the elements
            of inputs (or their size-adjusted variants)
        """
        assert isinstance(inputs, list)
        inputs = self.sz_align(inputs=inputs)
        return torch.cat(inputs, dim=1)


class NoAggregation(Aggregation):
    """
    No aggregation; only valid if there is just one input (if two are provided, the second one is dropped with a
    warning).
    """

    def __init__(self):
        super(NoAggregation, self).__init__()

    def forward(self, inputs: list):
        assert all([len(x.detach().size()) == 4 for x in inputs])
        if len(inputs) > 1:
            warnings.warn(
                "Provided more than one non-null inputs to NoAggregation class; only first input will be considered.",
                category=UserWarning)
        return inputs[0]


if __name__ == '__main__':
    import random
    for i in range(100):
        inputs = []
        print(f"\nROUND {i}:")
        n = random.randint(2, 5)
        for n in range(n):
            c = random.randint(1, 30)
            h = random.randint(1, 128)
            w = random.randint(1, 128)
            x = torch.ones(1, c, h, w)
            inputs.append(x)
            in_channels = [x.size(1) for x in inputs]

        print("X sizes:", [x.size() for x in inputs])

        add_pad_cst = Addition(in_channels, pad_or_interpolate='pad', pad_mode='constant')
        add_pad_ref = Addition(in_channels, pad_or_interpolate='pad', pad_mode='reflect')
        add_pad_rep = Addition(in_channels, pad_or_interpolate='pad', pad_mode='replicate')
        add_inter_near = Addition(in_channels, pad_or_interpolate='interpolate', interpolate_mode='nearest')
        add_inter_bilin = Addition(in_channels, pad_or_interpolate='interpolate', interpolate_mode='bilinear')

        cat_pad_cst = Concatenation(pad_or_interpolate='pad', pad_mode='constant')
        cat_pad_ref = Concatenation(pad_or_interpolate='pad', pad_mode='reflect')
        cat_pad_rep = Concatenation(pad_or_interpolate='pad', pad_mode='replicate')
        cat_inter_near = Concatenation(pad_or_interpolate='interpolate', interpolate_mode='nearest')
        cat_inter_bilin = Concatenation(pad_or_interpolate='interpolate', interpolate_mode='bilinear')


        print('\nadd_pad_cst', add_pad_cst(inputs).size())
        print('\nadd_pad_ref', add_pad_ref(inputs).size())
        print('\nadd_pad_rep', add_pad_rep(inputs).size())
        print('\nadd_inter_near', add_inter_near(inputs).size())
        print('\nadd_inter_bilin', add_inter_bilin(inputs).size())
        print('\ncat_pad_cst', cat_pad_cst(inputs).size())
        print('\ncat_pad_ref', cat_pad_ref(inputs).size())
        print('\ncat_pad_rep', cat_pad_rep(inputs).size())
        print('\ncat_inter_near', cat_inter_near(inputs).size())
        print('\ncat_inter_bilin', cat_inter_bilin(inputs).size())


