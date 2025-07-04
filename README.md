# ComfyUi-Load-Image-And-Display-Prompt-Metadata
This custom node displays the positive and negative prompts of a loaded ComfyUI image.
It additionally reads the seed, the number of sampling steps, and the CFG scale
from the image metadata so they can be viewed and reused in your workflow.

<img src="https://github.com/user-attachments/assets/ccd18d75-16df-4f69-9644-9822bc807535" width="500" />

## 1) Install
Navigate to the **ComfyUI/custom_nodes** folder and run the following command in your terminal:

```git clone https://github.com/BigStationW/ComfyUi-Load-Image-And-Display-Prompt-Metadata```

## 2) Usage
Double click on the empty space of ComfyUI's node interface and search for
"Load Image And Display Prompt Metadata".
The node will display the seed, steps and CFG scale in dedicated fields and
expose them as outputs so they can be connected to other nodes.
