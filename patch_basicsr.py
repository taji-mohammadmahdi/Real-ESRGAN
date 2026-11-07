from pathlib import Path
import basicsr

basicsr_path = Path(basicsr.__file__).parent
target_file = basicsr_path / "data" / "degradations.py"

content = target_file.read_text()

old = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
new = "from torchvision.transforms.functional import rgb_to_grayscale"

if old in content:
    content = content.replace(old, new)
    target_file.write_text(content)
    print("BasicSR patched successfully!")
elif new in content:
    print("BasicSR is already patched.")
else:
    raise RuntimeError("Could not find the expected import in degradations.py")
