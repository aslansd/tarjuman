# Examples

| script | what it shows |
|---|---|
| `01_run_neuroml_cell.py` | convert a NeuroML cell, run it, record voltage and gate variables |
| `02_fit_a_channel_parameter.py` | recover a `channelDensity` conductance by gradient descent through the converted model |
| `03_c302_connectome.py` | run a c302 *C. elegans* model: custom LEMS gates, calcium pools, gap junctions |

Both use the fixtures in `tests/data`, so they run without downloading
anything. Install the extras first:

```bash
pip install tarjuman[examples]
```

To run a c302 connectome model:

```bash
git clone https://github.com/openworm/c302
python examples/03_c302_connectome.py c302/examples/LEMS_c302_C_Oscillator.xml
```

To run a LEMS file from the NeuroML2 repository instead:

```bash
git clone https://github.com/NeuroML/NeuroML2
tarjuman run NeuroML2/LEMSexamples/LEMS_NML2_Ex25_MultiComp.xml --plot multicomp.png
```
