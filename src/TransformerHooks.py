"""
Gestion des forward hooks sur les couches Transformer de TRIBE v2.
Capture les activations des couches attention et feedforward du FmriEncoderModel
pendant le forward pass de model.predict().
"""
from collections import defaultdict
import torch


class TransformerHooks:
    """Attache des forward hooks sur les couches attention et FFN de FmriEncoderModel.

    Usage :
        hooks = TransformerHooks(fmri_enc)
        hooks.attacher()
        model.predict(events)
        features = hooks.get_features()
        hooks.retirer()
    """

    def __init__(self, fmri_enc):
        self.fmri_enc = fmri_enc
        self.features = defaultdict(list)
        self._hooks = []

    def _make_hook(self, name: str):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            self.features[name].append(out.detach().cpu())
        return hook

    def attacher(self) -> int:
        """Attache les hooks sur toutes les couches attention et FFN du Transformer.
        Retourne le nombre de hooks enregistrés.
        """
        self.features.clear()
        self._hooks.clear()

        # Each transformer layer: even indices = Attention, odd = FeedForward
        # encoder.layers[i][1] is the actual module
        for i, layer_block in enumerate(self.fmri_enc.encoder.layers):
            submodule = layer_block[1]
            layer_type = 'attn' if i % 2 == 0 else 'ffn'
            transformer_layer_idx = i // 2
            name = f'encoder.layer{transformer_layer_idx}.{layer_type}'
            self._hooks.append(
                submodule.register_forward_hook(self._make_hook(name))
            )

        print(f"Hooks enregistrés : {len(self._hooks)}")
        return len(self._hooks)

    def retirer(self) -> None:
        """Retire tous les hooks enregistrés."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_features(self) -> dict:
        """Retourne les activations capturées sous forme de dict {nom: array numpy}."""
        result = {}
        for layer_name, tensors in self.features.items():
            stacked = torch.cat(tensors, dim=0)
            safe_name = layer_name.replace('.', '_')
            result[safe_name] = stacked.numpy()
            print(f"  {layer_name:40s}  shape={tuple(stacked.shape)}")
        return result
