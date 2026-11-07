from pathlib import Path
import basicsr

basicsr_path = Path(basicsr.__file__).parent
target_file = basicsr_path / "data" / "degradations.py"

content = target_file.read_text()

old_imports = [
    "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
    "from torchvision.transforms._functional_tensor import rgb_to_grayscale",
]

new_import = "from torchvision.transforms.functional import rgb_to_grayscale"

if new_import in content:
    print("BasicSR is already patched.")

else:
    for old_import in old_imports:
        if old_import in content:
            content = content.replace(old_import, new_import)
            target_file.write_text(content)
            print("BasicSR patched successfully!")
            break
    else:
        raise RuntimeError("Could not find the expected import.")
