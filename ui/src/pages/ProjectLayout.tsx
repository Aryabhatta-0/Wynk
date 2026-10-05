import { NavLink, Outlet, useLocation, useOutletContext, useParams } from "react-router";
import { type Project, api } from "@/api";
import { FlowSteps, type FlowStep } from "@/components/app/FlowSteps";
import { ErrorState, LoadingState } from "@/components/app/ui";
import { useResource } from "@/lib/useResource";
import { cn } from "@/lib/utils";

export interface ProjectContext {
  project: Project;
  reloadProject: () => void;
}

export const useProject = () => useOutletContext<ProjectContext>();

function stepFor(path: string): FlowStep {
  if (/\/experiments\/new/.test(path)) return "Configure";
  if (/\/experiments\/[^/]+\/results/.test(path)) return "Compare";
  if (/\/experiments\/[^/]+/.test(path)) return "Optimize";
  if (/\/workflows/.test(path)) return "Champion";
  if (/\/experiments/.test(path)) return "Optimize";
  return "Dataset";
}

export function ProjectLayout() {
  const { projectId = "" } = useParams();
  const { pathname } = useLocation();
  const project = useResource(() => api().getProject(projectId), [projectId]);

  if (project.state === "loading") return <LoadingState label="Loading project" rows={3} />;
  if (project.state === "error") return <ErrorState title="Could not load this project" message={project.error} onRetry={project.reload} />;

  const p = project.data;
  const tabs = [
    { to: "datasets", label: "Datasets", count: p.datasetCount },
    { to: "experiments", label: "Experiments", count: p.experimentCount },
    { to: "workflows", label: "Workflows" },
  ];

  return (
    <div className="space-y-5">
      <div className="space-y-3 border-b border-line">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <nav aria-label="Breadcrumb" className="text-xs text-ink-soft">
            <NavLink to="/projects" className="hover:text-ink">
              Projects
            </NavLink>
            <span className="mx-1.5" aria-hidden="true">
              /
            </span>
            <span className="text-ink">{p.name}</span>
          </nav>
          <FlowSteps current={stepFor(pathname)} />
        </div>
        <div>
          <h1 className="text-2xl font-bold">{p.name}</h1>
          {p.description && <p className="mt-0.5 text-sm text-ink-soft">{p.description}</p>}
        </div>
        <nav className="-mb-px flex gap-1" aria-label="Project sections">
          {tabs.map((t) => (
            <NavLink
              key={t.to}
              to={t.to}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-1.5 border-b-2 px-3 py-2 text-[13px] font-medium",
                  isActive ? "border-magenta-ink text-ink" : "border-transparent text-ink-soft hover:text-ink",
                )
              }
            >
              {t.label}
              {t.count !== undefined && t.count !== null && <span className="rounded-full bg-wash px-1.5 text-[11px] tnum text-ink-soft">{t.count}</span>}
            </NavLink>
          ))}
        </nav>
      </div>
      <Outlet context={{ project: p, reloadProject: project.reload } satisfies ProjectContext} />
    </div>
  );
}
