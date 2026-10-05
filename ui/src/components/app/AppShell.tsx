import { ChatsCircle, Database, FolderSimple, GithubLogo, Flask, TreeStructure } from "@phosphor-icons/react";
import { createContext, useContext, useMemo } from "react";
import { NavLink, Outlet, useMatch } from "react-router";
import { api } from "@/api";
import { GhostLogo } from "@/components/GhostFigure";
import { useResource } from "@/lib/useResource";
import { cn } from "@/lib/utils";

const REPO = "https://github.com/Aryabhatta-0/Wynk";

/** Lets screens that change projects (create, add a dataset) refresh the sidebar. */
const ShellContext = createContext<{ refreshProjects: () => void }>({ refreshProjects: () => {} });
export const useShell = () => useContext(ShellContext);

export function AppShell() {
  const projects = useResource(() => api().listProjects(), []);
  const inProject = useMatch("/projects/:projectId/*");
  const projectId = inProject?.params.projectId;
  const shell = useMemo(() => ({ refreshProjects: projects.reload }), [projects.reload]);

  return (
    <ShellContext.Provider value={shell}>
      <div className="flex min-h-dvh bg-white text-[14px] text-ink">
        <aside className="sticky top-0 hidden h-dvh w-[232px] shrink-0 flex-col border-r border-line bg-white lg:flex" aria-label="Main">
          <NavLink to="/projects" className="flex h-14 shrink-0 items-center gap-2 border-b border-line px-4" aria-label="wynk home">
            <GhostLogo className="h-6 w-auto" />
            <span className="font-display text-lg font-bold tracking-tight">wynk</span>
          </NavLink>

          <nav className="min-h-0 flex-1 overflow-y-auto px-2 py-3 text-[13px]">
            <SideLink to="/projects" end icon={FolderSimple}>
              Projects
            </SideLink>
            {projects.state === "ready" && projects.data.length > 0 && (
              <ul className="mt-1 mb-2 space-y-0.5 pl-3" aria-label="Your projects">
                {projects.data.map((p) => (
                  <li key={p.id}>
                    <NavLink
                      to={`/projects/${p.id}`}
                      className={({ isActive }) =>
                        cn("block truncate rounded-md px-2.5 py-1.5 text-ink-soft hover:bg-wash hover:text-ink", isActive && "font-semibold text-ink")
                      }
                    >
                      {p.name}
                    </NavLink>
                    {projectId === p.id && (
                      <ul className="mt-0.5 mb-1 space-y-0.5 border-l border-line pl-2 ml-2.5">
                        <li>
                          <SideLink to={`/projects/${p.id}/datasets`} icon={Database} small>
                            Datasets
                          </SideLink>
                        </li>
                        <li>
                          <SideLink to={`/projects/${p.id}/experiments`} icon={Flask} small>
                            Experiments
                          </SideLink>
                        </li>
                        <li>
                          <SideLink to={`/projects/${p.id}/workflows`} icon={TreeStructure} small>
                            Workflows
                          </SideLink>
                        </li>
                      </ul>
                    )}
                  </li>
                ))}
              </ul>
            )}
            <div className="mt-4 border-t border-line pt-3">
              <SideLink to="/playground" icon={ChatsCircle}>
                Ask playground
              </SideLink>
            </div>
          </nav>

          <div className="space-y-2 border-t border-line p-3">
            <DataSourceBadge />
            <a href={REPO} className="flex items-center gap-1.5 text-xs text-ink-soft hover:text-ink" target="_blank" rel="noreferrer">
              <GithubLogo size={14} aria-hidden="true" /> Source on GitHub
            </a>
          </div>
        </aside>

        <div className="flex min-w-0 flex-1 flex-col">
          <header className="flex h-12 items-center justify-between gap-3 border-b border-line px-4 lg:hidden">
            <NavLink to="/projects" className="flex items-center gap-2" aria-label="wynk home">
              <GhostLogo className="h-5 w-auto" />
              <span className="font-display text-base font-bold">wynk</span>
            </NavLink>
            <div className="flex items-center gap-3 text-[13px]">
              <NavLink to="/projects" className="text-ink-soft hover:text-ink">
                Projects
              </NavLink>
              <DataSourceBadge compact />
            </div>
          </header>
          <main className="mx-auto w-full max-w-[1480px] min-w-0 flex-1 px-4 py-5 sm:px-6 xl:px-8">
            <Outlet />
          </main>
        </div>
      </div>
    </ShellContext.Provider>
  );
}

function SideLink({
  to,
  icon: IconCmp,
  children,
  end,
  small,
}: {
  to: string;
  icon: typeof Database;
  children: React.ReactNode;
  end?: boolean;
  small?: boolean;
}) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        cn(
          "flex items-center gap-2 rounded-md px-2.5 py-1.5 text-ink-soft hover:bg-wash hover:text-ink",
          small && "py-1 text-[12.5px]",
          isActive && "bg-magenta-wash font-semibold text-magenta-ink hover:bg-magenta-wash hover:text-magenta-ink",
        )
      }
    >
      <IconCmp size={small ? 14 : 16} aria-hidden="true" />
      {children}
    </NavLink>
  );
}

/** Always visible: whether the numbers on screen are mock data or the live product API. */
export function DataSourceBadge({ compact }: { compact?: boolean }) {
  const mock = api().mode === "mock";
  return (
    <div
      data-testid="data-source"
      title={mock ? "Mock adapter: invented data, simulated runs, resets on reload." : "Live product API"}
      className={cn(
        "rounded-lg border px-2.5 py-1.5 text-xs",
        mock ? "border-dashed border-warn/50 bg-warn-wash text-warn" : "border-line-strong text-ink-soft",
        compact && "px-2 py-0.5",
      )}
    >
      <span className="font-semibold">{mock ? "Mock data" : "Live API"}</span>
      {!compact && <span className="block text-[11px] leading-snug opacity-90">{mock ? "Simulated runs. Resets on reload." : "Product API"}</span>}
    </div>
  );
}
