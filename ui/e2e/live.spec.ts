import { type APIRequestContext, type Page, expect, test } from "@playwright/test";
import { createHash } from "node:crypto";

/*
  Live mode against a real product API (`python -m api.product`, started by playwright.config.ts on
  a fresh data directory). Nothing here is mocked: every number on screen came from the server.
*/
const API = "http://127.0.0.1:8790/api/v1";
// #32: direct API calls carry the dev workspace's key; the browser itself never holds it (the dev
// proxy adds it), which these flows also prove
const AUTH = { Authorization: `Bearer ${process.env.WYNK_E2E_API_KEY ?? `wynk_sk_0000000000000e2e_${"E".repeat(43)}`}` };

const CSV = `ticket_id,subject,body,customer_tier,queue
T-1,Charged twice,"My card shows two charges, both for March.",pro,billing
T-2,Cannot log in,SSO redirects back to the login page.,enterprise,technical
T-3,Update company name,We rebranded; please change the account name.,,account
T-4,Where is my order,Tracking has not moved for a week.,free,shipping
T-5,Refund request,"Downgraded mid-cycle, expected a prorated refund.",pro,billing
T-6,Webhook failures,Deliveries to our endpoint time out since Monday.,enterprise,technical
T-7,Add a teammate,How do I invite a colleague to our workspace?,free,account
T-8,Damaged parcel,The box arrived crushed and the device is cracked.,pro,shipping
`;
const sha256 = (s: string | Buffer) => createHash("sha256").update(s).digest("hex");
const unique = () => `${Date.now().toString(36)}-${Math.floor(Math.random() * 1e6).toString(36)}`;

/** Fails the test if the browser ever loads mock adapter code. */
function watchForMockCode(page: Page) {
  const loaded: string[] = [];
  page.on("request", (r) => {
    if (/\/src\/api\/mock\//.test(r.url())) loaded.push(r.url());
  });
  return loaded;
}

async function apiProject(request: APIRequestContext, name: string): Promise<string> {
  const res = await request.post(`${API}/projects`, { data: { name, description: "" }, headers: AUTH });
  expect(res.status()).toBe(201);
  return (await res.json()).project_id;
}

async function upload(page: Page, name: string, content: string | Buffer) {
  await page.getByTestId("dataset-file").setInputFiles({ name, mimeType: "application/octet-stream", buffer: Buffer.from(content) });
}

test("live: create project → upload CSV → inspect → map → register → split → reload → data persists", async ({ page, request }) => {
  const mockCode = watchForMockCode(page);
  const projectName = `E2E live ${unique()}`;
  const datasetId = `tickets-${unique()}`;

  await page.goto("/projects");
  await expect(page.getByTestId("data-source").first()).toContainText("Live API");

  // Project
  // a fresh backend has no projects, so the empty state offers the same button
  await page.getByRole("button", { name: "New project" }).first().click();
  await page.getByLabel("Name").fill(projectName);
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByRole("heading", { name: projectName })).toBeVisible();
  await expect(page.getByText("No datasets in this project")).toBeVisible();

  // Upload → Inspect: the server reads every row
  await page.getByRole("link", { name: "Add dataset" }).click();
  await upload(page, "tickets.csv", CSV);
  await expect(page).toHaveURL(/\/datasets\/new\?upload=u-[0-9a-f]+/);
  await expect(page.getByTestId("inspection-rows")).toHaveText("8");
  await expect(page.getByTestId("inspection-hash")).toHaveText(sha256(CSV));
  const tier = page.getByRole("table", { name: "Columns and roles" }).getByRole("row").filter({ hasText: "customer_tier" });
  await expect(tier).toContainText("yes"); // nullable: one row has no value
  await expect(page.getByRole("table", { name: "Preview rows" }).getByRole("row")).toHaveCount(9);

  // Map columns
  await expect(page.getByLabel("Role of queue")).toHaveValue("target");
  await expect(page.getByLabel("Role of ticket_id")).toHaveValue("id");
  await page.getByLabel("Role of customer_tier").selectOption("context");

  // Register
  await page.getByLabel("Dataset id").fill(datasetId);
  await page.getByRole("button", { name: "Register dataset" }).click();
  await expect(page).toHaveURL(new RegExp(`/datasets/${datasetId}\\?version=1$`));
  await expect(page.getByTestId("version-rows")).toHaveText("8");
  await expect(page.getByTestId("version-content-hash")).toHaveText(sha256(CSV));
  const identity = (await page.getByTestId("version-identity-hash").textContent())!;
  expect(identity).toMatch(/^[0-9a-f]{64}$/);
  await expect(page.getByText("No splits for this version yet.")).toBeVisible();

  // Create splits: 8 rows, 25% validation, 25% test -> 4 / 2 / 2 (server-computed)
  await page.getByLabel("Validation (%)").fill("25");
  await page.getByLabel("Test (%)").fill("25");
  await page.getByLabel("Seed").fill("7");
  await page.getByRole("button", { name: "Create splits" }).click();
  const row = page.getByTestId("splits-row");
  await expect(row).toHaveCount(1);
  const splitsHash = (await row.getAttribute("data-splits-hash"))!;
  expect(splitsHash).toMatch(/^[0-9a-f]{64}$/);
  await expect(row.getByTestId("size-optimization")).toHaveText("4");
  await expect(row.getByTestId("size-validation")).toHaveText("2");
  await expect(row.getByTestId("size-test")).toHaveText("2");

  // Reload: everything comes back from the backend
  await page.reload();
  await expect(page.getByTestId("version-identity-hash")).toHaveText(identity);
  await expect(page.getByTestId("splits-row")).toHaveAttribute("data-splits-hash", splitsHash);
  await expect(page.getByTestId("splits-row").getByTestId("size-optimization")).toHaveText("4");

  await page.goto("/projects");
  await page.getByRole("table", { name: "Projects" }).getByRole("link", { name: projectName }).click();
  await expect(page.getByTestId("dataset-row").filter({ hasText: datasetId })).toBeVisible();

  // and the backend itself holds it
  const stored = await (await request.get(`${API}/datasets/${datasetId}/versions/1/splits/${splitsHash}`, { headers: AUTH })).json();
  expect(stored.sizes).toEqual({ optimization: 4, test: 2, validation: 2 });
  expect(stored.splits.dataset_hash).toBe(identity);

  expect(mockCode).toEqual([]);
});

test("live: backend validation errors are shown with the server's reason and code", async ({ page, request }) => {
  const mockCode = watchForMockCode(page);
  const pid = await apiProject(request, `E2E errors ${unique()}`);
  await page.goto(`/projects/${pid}/datasets/new`);

  const expectError = async (code: string, title: string, reason: RegExp, timeout?: number) => {
    const alert = page.getByRole("alert").filter({ hasText: title });
    await expect(alert).toBeVisible({ timeout });
    await expect(alert).toContainText(reason);
    await expect(alert.getByTestId("error-code")).toHaveText(code);
  };

  await upload(page, "bad.csv", "ticket_id,subject\nT-1\n");
  await expectError("malformed_csv", "The CSV file is malformed", /line 2 has 1 fields, the header has 2/);

  await upload(page, "rows.parquet", "PAR1");
  await expectError("unsupported_format", "Unsupported file format", /format must be one of: csv, jsonl/);

  await upload(page, "big.csv", Buffer.alloc(1024 * 1024 + 1, "a"));
  // refused before the server reads the body; the error must still reach the browser (not a network error)
  await expectError("payload_too_large", "The file is too large", /larger than 1048576 bytes/, 20_000);

  // duplicate ids pass inspection, and registration with that id column is refused
  await upload(page, "dupes.csv", "ticket_id,subject,queue\nT-1,a,billing\nT-1,b,account\n");
  await expect(page.getByTestId("inspection-rows")).toHaveText("2");
  await expect(page.getByLabel("Role of ticket_id")).toHaveValue("id");
  await page.getByLabel("Dataset id").fill(`dupes-${unique()}`);
  await page.getByRole("button", { name: "Register dataset" }).click();
  await expectError("duplicate_row_id", "Row ids are not unique", /row id 'T-1' appears more than once/);

  // a dataset id owned by another project
  const other = await apiProject(request, `E2E other ${unique()}`);
  const taken = `taken-${unique()}`;
  const up = await request.post(`${API}/projects/${other}/uploads?format=csv&filename=t.csv`, {
    data: Buffer.from(CSV),
    headers: { "Content-Type": "application/octet-stream", ...AUTH },
  });
  const reg = await request.post(`${API}/uploads/${(await up.json()).upload_id}/register`, {
    data: { dataset_id: taken, name: "t", input_columns: ["subject"], target_columns: ["queue"], row_ids: "generated" },
    headers: AUTH,
  });
  expect(reg.status()).toBe(201);
  await page.getByLabel("Role of ticket_id").selectOption("ignore"); // generated ids: no duplicate problem
  await page.getByLabel("Dataset id").fill(taken);
  await page.getByRole("button", { name: "Register dataset" }).click();
  await expectError("dataset_conflict", "This dataset id is already taken", /belongs to another project/);

  expect(mockCode).toEqual([]);
});

test("live: experiment screens say the experiment backend does not exist yet", async ({ page, request }) => {
  const mockCode = watchForMockCode(page);
  const apiCalls: string[] = [];
  page.on("request", (r) => {
    const path = new URL(r.url()).pathname;
    if (path.startsWith("/api/")) apiCalls.push(path);
  });
  const pid = await apiProject(request, `E2E experiments ${unique()}`);
  for (const path of ["experiments", "experiments/new", "workflows"]) {
    await page.goto(`/projects/${pid}/${path}`);
    await expect(page.getByTestId("experiment-backend-required")).toContainText("not built yet");
    await expect(page.getByRole("link", { name: "#23" })).toBeVisible();
    await expect(page.getByTestId("simulated-notice")).toHaveCount(0);
  }
  expect(apiCalls.filter((p) => !/^\/api\/v1\/projects(\/[^/]+)?$/.test(p))).toEqual([]);
  expect(mockCode).toEqual([]);
});
