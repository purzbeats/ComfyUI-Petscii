import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// `.petv` is not an image, an audio file or a video, so the frontend has nothing
// to do with the save node's result on its own — it would report success and show
// nothing. This adds the one affordance the file actually wants: a link to it.
//
// The node also returns the summary as `text`, which the frontend renders without
// any help from here, so a client that never loads this script still sees where
// the stream went.

const NODE = "PETSCIISavePETV";
const WIDGET = "petv_download";

function link(node, file) {
  const url = api.apiURL(
    `/view?filename=${encodeURIComponent(file.filename)}` +
      `&subfolder=${encodeURIComponent(file.subfolder ?? "")}` +
      `&type=${encodeURIComponent(file.type ?? "output")}`
  );

  let widget = node.widgets?.find((w) => w.name === WIDGET);
  if (!widget) {
    const anchor = document.createElement("a");
    anchor.style.cssText =
      "display:block;padding:4px 8px;font-family:monospace;font-size:11px;" +
      "text-align:center;color:#8bd;text-decoration:none;overflow:hidden;" +
      "text-overflow:ellipsis;white-space:nowrap";
    anchor.download = "";
    widget = node.addDOMWidget(WIDGET, "petv", anchor, { serialize: false });
    widget.element = anchor;
  }

  widget.element.href = url;
  widget.element.textContent = `⤓ ${file.filename}`;
  widget.element.title = file.filename;
}

app.registerExtension({
  name: "petscii.save-petv",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const files = message?.petv;
      if (Array.isArray(files) && files.length) {
        link(this, files[files.length - 1]);
      }
    };
  },
});
