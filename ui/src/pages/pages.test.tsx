import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RouterProvider, createMemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";
import { setApi } from "@/api";
import { createMockApi } from "@/api/mock/mockApi";
import { routes } from "@/router";

function renderAt(path: string, options: Parameters<typeof createMockApi>[0] = {}) {
  setApi(createMockApi({ latencyMs: 0, ...options }));
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  render(<RouterProvider router={router} />);
  return router;
}

describe("screens on the mock adapter", () => {
  it("lists projects and always labels the data as mock", async () => {
    renderAt("/projects");
    const table = await screen.findByRole("table", { name: "Projects" }, { timeout: 3000 });
    expect(within(table).getByRole("link", { name: "Support triage" })).toHaveAttribute("href", "/projects/p-support");
    expect(within(table).getByRole("link", { name: "Invoice extraction" })).toBeInTheDocument();
    expect(screen.getAllByTestId("data-source")[0]).toHaveTextContent("Mock data");
  });

  it("shows an error state with retry when the adapter fails", async () => {
    renderAt("/projects", { failOn: ["listProjects"] });
    const alert = await screen.findByRole("alert", {}, { timeout: 3000 });
    expect(alert).toHaveTextContent("Could not load projects");
    expect(within(alert).getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });

  it("separates hard constraints from the optimization preference on the configure screen", async () => {
    renderAt("/projects/p-support/experiments/new?dataset=d-tickets");
    expect(await screen.findByText("Hard constraints", {}, { timeout: 3000 })).toBeInTheDocument();
    expect(screen.getByText("Optimization preference")).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /Classification/ })).toBeChecked();
    expect(screen.getByText("billing")).toBeInTheDocument();
    // a broken split is caught before anything is sent
    const user = userEvent.setup();
    const test = screen.getByLabelText("Held-out test (%)");
    await user.clear(test);
    await user.type(test, "40");
    await user.click(screen.getByRole("button", { name: "Start optimization" }));
    expect(await screen.findByText("Splits must add up to 100%.")).toBeInTheDocument();
  });

  it("compares the three methods on all splits for a finished run, with measured facts and no deployment", async () => {
    renderAt("/projects/p-support/experiments/e-triage-1/results");
    const table = await screen.findByRole("table", { name: "Method comparison" }, { timeout: 3000 });
    for (const name of ["Fixed baseline", "Random search", "Wynk (ACO)"]) expect(within(table).getByText(name)).toBeInTheDocument();
    expect(within(table).queryByText("locked")).not.toBeInTheDocument();
    const facts = screen.getByTestId("why-facts");
    expect(facts).toHaveTextContent("Held-out test accuracy");
    expect(facts).toHaveTextContent("for fixed baseline");
    expect(screen.getByRole("button", { name: /Deploy champion/ })).toBeDisabled();
    expect(screen.getByText("Deployment is coming soon")).toBeInTheDocument();
  });

  it("shows a failed run's reason", async () => {
    renderAt("/projects/p-invoices/experiments/e-invoices-1");
    expect(await screen.findByText("The run failed", {}, { timeout: 3000 })).toBeInTheDocument();
    expect(screen.getByText(/HTTP 503/)).toBeInTheDocument();
  });
});
