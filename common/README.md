# Shared TinyML helpers

Import from project training scripts via:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.quantize_utils import convert_to_int8_tflite, tflite_to_c_array
```
