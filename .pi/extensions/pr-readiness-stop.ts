import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const READINESS_HOOK = resolve(REPO_ROOT, "agent/hooks/pr-readiness-stop.sh");

export default function (pi: ExtensionAPI) {
	let lastReport = "";

	pi.on("agent_settled", async (_event, ctx) => {
		if (ctx.mode !== "tui") return;

		const result = await pi.exec("bash", [READINESS_HOOK], {
			cwd: ctx.cwd,
			timeout: 30_000,
		});
		const report = result.stderr.trim();
		if (report === lastReport) return;
		if (result.code === 0 && report.startsWith("WARNING:")) {
			lastReport = report;
			ctx.ui.notify(report, "warning");
			return;
		}
		if (result.code !== 2 || !report) {
			lastReport = "";
			return;
		}

		lastReport = report;
		pi.sendUserMessage(report);
	});
}
