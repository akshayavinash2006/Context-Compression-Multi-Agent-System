import { motion } from "framer-motion";
import { Page, rise, stagger } from "../components/ui";
import { ZONES } from "../layout";
import { STATUS_COLOR, STATUS_LABEL } from "../status";
import { useStore } from "../store";

const ROLE_OF = (id: string) => (["history", "diagnosis", "report"].includes(id) ? id : "sensor");

export function Agents() {
  const { agents, models, zonesInRun, set } = useStore();
  return (
    <Page title="Agents" sub="One agent per zone. Production zones are Sensor agents; the rest support them.">
      <motion.div className="agent-grid" variants={stagger}>
        {ZONES.map((z) => {
          const a = agents[z.id];
          const st = a?.status ?? "idle";
          const inRun = zonesInRun.length === 0 || z.kind === "support" || zonesInRun.includes(z.id);
          const model = models[ROLE_OF(z.id)]?.primary;
          return (
            <motion.article key={z.id} variants={rise} className={`agent-card ${inRun ? "" : "muted"}`} style={{ ["--accent" as string]: z.color }} whileHover={{ y: -4 }}
              onClick={() => { set({ selectedAgent: z.id }); location.hash = "#/overview"; }}>
              <div className="ac-top">
                <span className="ac-badge">{z.kind === "support" ? "Support" : "Production"}</span>
                <span className="pill" style={{ ["--c" as string]: STATUS_COLOR[st] }}><i />{STATUS_LABEL[st]}</span>
              </div>
              <h3>{z.name}</h3>
              <p className="muted">{z.subtitle}</p>
              <div className="mini-stats">
                <div><b>{a?.writes ?? 0}</b><span>writes</span></div>
                <div><b>{a?.reads ?? 0}</b><span>reads</span></div>
                <div><b>{(a?.inputTokens ?? 0).toLocaleString()}</b><span>tokens</span></div>
              </div>
              <div className="ac-model">{inRun ? (model ?? "No model assigned yet") : "No machines in this scenario"}</div>
            </motion.article>
          );
        })}
      </motion.div>
    </Page>
  );
}
