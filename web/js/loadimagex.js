import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Configuration option to enable/disable logging
const ENABLE_LOGGING = false; // Set to true to enable all console logs

// Helper function for conditional logging
function log(message, style = "") {
    if (ENABLE_LOGGING) {
        if (style) {
            console.log(message, style);
        } else {
            console.log(message);
        }
    }
}

function logError(message, error) {
    if (ENABLE_LOGGING) {
        console.error(message, error);
    }
}

// Helper function to clean potential non-standard JSON from metadata
function cleanJSONString(jsonString) {
    if (!jsonString) return null;
    return jsonString.replace(/:\s*NaN/g, ': null');
}

// Self-contained function to parse metadata from a PNG file's raw data.
function parsePNGMetadata(arrayBuffer) {
    const dataView = new DataView(arrayBuffer);
    const metadata = {};

    // Check for PNG signature
    const pngSignature = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    for (let i = 0; i < 8; i++) {
        if (dataView.getUint8(i) !== pngSignature[i]) {
            logError("[LoadImageX] Not a valid PNG file.");
            return null;
        }
    }

    let offset = 8;
    while (offset < dataView.byteLength) {
        const length = dataView.getUint32(offset);
        offset += 4;
        const chunkType = String.fromCharCode(
            dataView.getUint8(offset), dataView.getUint8(offset + 1),
            dataView.getUint8(offset + 2), dataView.getUint8(offset + 3)
        );
        offset += 4;

        if (chunkType === 'tEXt' || chunkType === 'iTXt') {
            const chunkData = new Uint8Array(arrayBuffer, offset, length);
            const text = new TextDecoder('utf-8').decode(chunkData);
            const nullIndex = text.indexOf('\0');
            if (nullIndex !== -1) {
                const keyword = text.substring(0, nullIndex);
                const value = text.substring(nullIndex + 1);
                metadata[keyword] = value;
            }
        }

        offset += length + 4; // Skip data and CRC
        if (chunkType === 'IEND') break;
    }
    return metadata;
}

// Helper function to find positive/negative prompts in the workflow
function extractPromptsFromWorkflow(workflow) {
    const prompts = { positive: "", negative: "", seed: null, steps: null, cfg: null };
    if (!workflow) return prompts;
    
    try {
        // First, find nodes that connect to positive/negative inputs
        let positiveNodeId = null;
        let negativeNodeId = null;
        
        // Check KSampler nodes first
        for (const nodeId in workflow) {
            const node = workflow[nodeId];

            if ((node.class_type === "KSampler" || node.class_type === "KSamplerAdvanced") && node.inputs) {
                if (node.inputs.positive && Array.isArray(node.inputs.positive)) {
                    positiveNodeId = String(node.inputs.positive[0]);
                }
                if (node.inputs.negative && Array.isArray(node.inputs.negative)) {
                    negativeNodeId = String(node.inputs.negative[0]);
                }

                const seedVal = node.inputs.seed;
                if (prompts.seed === null && (typeof seedVal === "number" || (typeof seedVal === "string" && !isNaN(parseInt(seedVal))))) {
                    prompts.seed = parseInt(seedVal);
                }

                const stepsVal = node.inputs.steps;
                if (prompts.steps === null && (typeof stepsVal === "number" || (typeof stepsVal === "string" && !isNaN(parseInt(stepsVal))))) {
                    prompts.steps = parseInt(stepsVal);
                }

                const cfgVal = node.inputs.cfg;
                if (prompts.cfg === null && (typeof cfgVal === "number" || (typeof cfgVal === "string" && !isNaN(parseFloat(cfgVal))))) {
                    prompts.cfg = parseFloat(cfgVal);
                }
            }
        }
        
        // Check for AdaptiveGuidance nodes
        if (!positiveNodeId && !negativeNodeId) {
            for (const nodeId in workflow) {
                const node = workflow[nodeId];
                
                if (node.class_type === "AdaptiveGuidance" && node.inputs) {
                    if (node.inputs.positive && Array.isArray(node.inputs.positive)) {
                        positiveNodeId = String(node.inputs.positive[0]);
                    }
                    if (node.inputs.negative && Array.isArray(node.inputs.negative)) {
                        negativeNodeId = String(node.inputs.negative[0]);
                    }
                }
            }
        }
        
        // If no KSampler or AdaptiveGuidance, check for CFGGuider
        if (!positiveNodeId && !negativeNodeId) {
            for (const nodeId in workflow) {
                const node = workflow[nodeId];
                
                if (node.class_type === "CFGGuider" && node.inputs) {
                    if (node.inputs.positive && Array.isArray(node.inputs.positive)) {
                        let posId = node.inputs.positive[0];
                        const posNode = workflow[posId];
                        if (posNode && posNode.class_type === "ChromaPaddingRemoval" && 
                            posNode.inputs && posNode.inputs.conditioning) {
                            positiveNodeId = String(posNode.inputs.conditioning[0]);
                        } else {
                            positiveNodeId = String(posId);
                        }
                    }
                    
                    if (node.inputs.negative && Array.isArray(node.inputs.negative)) {
                        let negId = node.inputs.negative[0];
                        const negNode = workflow[negId];
                        if (negNode && negNode.class_type === "ChromaPaddingRemoval" && 
                            negNode.inputs && negNode.inputs.conditioning) {
                            negativeNodeId = String(negNode.inputs.conditioning[0]);
                        } else {
                            negativeNodeId = String(negId);
                        }
                    }
                }
                
                // Also check for NAGCFGGuider
                if (node.class_type === "NAGCFGGuider" && node.inputs) {
                    if (node.inputs.positive && Array.isArray(node.inputs.positive)) {
                        positiveNodeId = String(node.inputs.positive[0]);
                    }
                    if (node.inputs.nag_negative && Array.isArray(node.inputs.nag_negative)) {
                        negativeNodeId = String(node.inputs.nag_negative[0]);
                    }
                }
            }
        }
        
        // Now extract the actual prompt texts
        for (const nodeId in workflow) {
            const node = workflow[nodeId];
            
            if (node.class_type === "CLIPTextEncode" && node.inputs && node.inputs.text) {
                const text = node.inputs.text;
                
                if (positiveNodeId && String(nodeId) === positiveNodeId) {
                    prompts.positive = text;
                } else if (negativeNodeId && String(nodeId) === negativeNodeId) {
                    prompts.negative = text;
                } else {
                    const title = node._meta?.title?.toLowerCase() || "";
                    if (title.includes("negative") || title.includes("nag")) {
                        if (!prompts.negative) prompts.negative = text;
                    } else {
                        if (!prompts.positive) prompts.positive = text;
                    }
                }
            }
            
            // Handle CLIPTextEncodeFlux nodes
            else if (node.class_type === "CLIPTextEncodeFlux" && node.inputs) {
                const text = node.inputs.clip_l || node.inputs.t5xxl || "";
                
                if (positiveNodeId && String(nodeId) === positiveNodeId) {
                    prompts.positive = text;
                } else if (negativeNodeId && String(nodeId) === negativeNodeId) {
                    prompts.negative = text;
                }
            }
        }
    } catch (error) {
        logError("[LoadImageX] Error extracting prompts from workflow:", error);
    }
    return prompts;
}

// Main function to get metadata from an image and update the text boxes
async function updatePromptsFromImage(filename, node) {
    const positiveWidget = node.widgets.find(w => w.name === "positive_prompt");
    const negativeWidget = node.widgets.find(w => w.name === "negative_prompt");
    const seedWidget = node.widgets.find(w => w.name === "seed");
    const stepsWidget = node.widgets.find(w => w.name === "steps");
    const cfgWidget = node.widgets.find(w => w.name === "cfg");

    if (positiveWidget) positiveWidget.value = "";
    if (negativeWidget) negativeWidget.value = "";
    if (seedWidget) seedWidget.value = 0;
    if (stepsWidget) stepsWidget.value = 0;
    if (cfgWidget) cfgWidget.value = 0;

    try {
        const res = await api.fetchApi(`/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=`);
        if (!res.ok) throw new Error(`Failed to fetch image: ${res.status}`);
        
        const buffer = await res.arrayBuffer();
        const metadata = parsePNGMetadata(buffer);
        
        // Log metadata in blue
        log("%c[LoadImageX] Metadata:", "color: #0066ff; font-weight: bold");
        log("%c" + JSON.stringify(metadata, null, 2), "color: #0066ff");
        
        if (metadata && metadata.prompt) {
            const promptData = JSON.parse(cleanJSONString(metadata.prompt));
            const prompts = extractPromptsFromWorkflow(promptData);

            // Log positive prompt in green
            if (prompts.positive) {
                log("%c[LoadImageX] Positive Prompt:", "color: #00cc00; font-weight: bold");
                log("%c" + prompts.positive, "color: #00cc00");
            }
            
            // Log negative prompt in red
            if (prompts.negative) {
                log("%c[LoadImageX] Negative Prompt:", "color: #ff0000; font-weight: bold");
                log("%c" + prompts.negative, "color: #ff0000");
            }

            if (prompts.seed !== null) {
                log("%c[LoadImageX] Seed: " + prompts.seed, "color: #6a5acd");
            }
            if (prompts.steps !== null) {
                log("%c[LoadImageX] Steps: " + prompts.steps, "color: #6a5acd");
            }
            if (prompts.cfg !== null) {
                log("%c[LoadImageX] CFG: " + prompts.cfg, "color: #6a5acd");
            }

            if (positiveWidget && prompts.positive) positiveWidget.value = prompts.positive;
            if (negativeWidget && prompts.negative) negativeWidget.value = prompts.negative;
            if (seedWidget && prompts.seed !== null) seedWidget.value = prompts.seed;
            if (stepsWidget && prompts.steps !== null) stepsWidget.value = prompts.steps;
            if (cfgWidget && prompts.cfg !== null) cfgWidget.value = prompts.cfg;
        }
    } catch (error) {
        logError("[LoadImageX] Error processing image metadata:", error);
    }
}

app.registerExtension({
    name: "testt.LoadImageX",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LoadImageX") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                const self = this;
                
                const imageWidget = this.widgets.find(w => w.name === "image");
                if (!imageWidget) return r;

                // Store the original callback
                const originalCallback = imageWidget.callback;
                
                // Override the callback to handle both value change and preview update
                imageWidget.callback = function(value) {
                    if (value) {
                        // Update prompts from metadata
                        updatePromptsFromImage(value, self);
                        
                        // Trigger the node to update its outputs
                        // This is important for the preview to update
                        if (self.graph) {
                            self.graph.runStep();
                        }
                    }
                    
                    // Call original callback if it exists
                    if (originalCallback) {
                        return originalCallback.apply(this, arguments);
                    }
                };

                // If there's already a default image value, load its metadata
                if (imageWidget.value) {
                    // Use setTimeout to ensure the node is fully initialized
                    setTimeout(() => {
                        updatePromptsFromImage(imageWidget.value, self);
                    }, 100);
                }

                // Handle the upload button
                const uploadWidget = this.widgets.find(w => w.type === "button");
                if (uploadWidget) {
                    uploadWidget.callback = () => {
                        const fileInput = document.createElement("input");
                        fileInput.type = "file";
                        fileInput.accept = "image/png,image/jpeg,image/webp";
                        fileInput.style.display = "none";
                        document.body.appendChild(fileInput);

                        fileInput.onchange = async (e) => {
                            if (!e.target.files.length) {
                                document.body.removeChild(fileInput);
                                return;
                            }
                            const file = e.target.files[0];
                            const formData = new FormData();
                            formData.append("image", file);
                            formData.append("overwrite", "true");
                            
                            try {
                                const response = await api.fetchApi("/upload/image", { 
                                    method: "POST", 
                                    body: formData 
                                });
                                if (response.ok) {
                                    const data = await response.json();
                                    // Update the widget value
                                    imageWidget.value = data.name;
                                    // Trigger the callback to update prompts and preview
                                    if (imageWidget.callback) {
                                        imageWidget.callback(data.name);
                                    }
                                } else {
                                    logError("[LoadImageX] Upload failed:", response.statusText);
                                }
                            } catch (error) {
                                logError("[LoadImageX] Upload error:", error);
                            } finally {
                                document.body.removeChild(fileInput);
                            }
                        };
                        fileInput.click();
                    };
                }
                
                return r;
            };
        }
    }
});
