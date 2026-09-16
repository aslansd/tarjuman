# Examples

| script | what it shows |
|---|---|
| `01_run_neuroml_cell.py` | convert a NeuroML cell, run it, record voltage and gate variables |
| `02_fit_a_channel_parameter.py` | recover a `channelDensity` conductance by gradient descent through the converted model |

Both use the fixtures in `tests/data`, so they run without downloading
anything. Install the extras first:

```bash
pip install "tarjuman[examples]"
```

To run a LEMS file from the NeuroML2 repository instead:

```bash
git clone https://github.com/NeuroML/NeuroML2
tarjuman run NeuroML2/LEMSexamples/LEMS_NML2_Ex25_MultiComp.xml --plot multicomp.png
```
