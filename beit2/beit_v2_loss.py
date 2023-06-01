import torch


class NoDynamicShapesMaskedCrossedEntropy(torch.nn.CrossEntropyLoss):
    def __init__(
            self,
            invis_mask_key: str = "invis_mask",
            **super_kwargs
    ):
        self._ignore_index = 100000
        super(NoDynamicShapesMaskedCrossedEntropy, self).__init__(ignore_index=self._ignore_index, **super_kwargs)
        self.invis_mask_key = invis_mask_key

    def __call__(self, pred, batch):
        invis_mask = pred[self.invis_mask_key]
        invis_loss = self.call_on_mask(pred, invis_mask)
        return {'loss': invis_loss}

    def mask_labels(self, labels, mask):
        input_dim = labels.shape
        addition_mask = ~mask * self._ignore_index
        return ((labels * mask) + addition_mask).reshape((input_dim[0] * input_dim[1],))

    def mask_pred(self, pred, mask):
        input_dim = pred.shape
        mask = mask.unsqueeze(2).repeat((1, 1, input_dim[-1]))
        addition_mask = ~mask * self._ignore_index
        return ((pred * mask) + addition_mask).reshape((input_dim[0] * input_dim[1], input_dim[2]))

    def call_on_mask(self, pred, mask):
    #     visible_tokenizer_labels = pred['tokenizer_labels'][mask]
    #     visible_model_outputs = pred['beit_model_output'][mask]
        visible_tokenizer_labels = self.mask_labels(pred['tokenizer_labels'], mask)
        visible_model_outputs = self.mask_pred(pred['beit_model_output'], mask)
        vis_loss = super(NoDynamicShapesMaskedCrossedEntropy, self).__call__(input=visible_model_outputs, target=visible_tokenizer_labels)
        return vis_loss


class MaskedCrossedEntropy(torch.nn.CrossEntropyLoss):
    def __init__(
            self,
            invis_mask_key: str = "invis_mask",
            **super_kwargs
    ):
        super(MaskedCrossedEntropy, self).__init__(**super_kwargs)
        self.invis_mask_key = invis_mask_key

    def __call__(self, pred, batch):
        invis_mask = pred[self.invis_mask_key]
        invis_loss = self.call_on_mask(pred, invis_mask)
        return {'loss': invis_loss}

    def call_on_mask(self, pred, mask):
        visible_tokenizer_labels = pred['tokenizer_labels'][mask]
        visible_model_outputs = pred['beit_model_output'][mask]
        vis_loss = super(MaskedCrossedEntropy, self).__call__(input=visible_model_outputs, target=visible_tokenizer_labels)
        return vis_loss

