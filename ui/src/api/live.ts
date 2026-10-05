import { ApiError, type WynkApi } from "./client";

const NOT_CONNECTED = "The Wynk product API is not available yet. Run the UI with mock data (remove ?api=live) to explore the flow.";

const fail = (): Promise<never> => Promise.reject(new ApiError("not_connected", NOT_CONNECTED));

/**
 * Placeholder for the real product API. Every method fails honestly until the backend
 * contract is final; nothing here invents endpoints or reports success it did not get.
 */
export function createLiveApi(): WynkApi {
  return {
    mode: "live",
    listProjects: fail,
    getProject: fail,
    createProject: fail,
    listDatasets: fail,
    getDataset: fail,
    createDataset: fail,
    updateMapping: fail,
    listModels: fail,
    listExperiments: fail,
    getExperiment: fail,
    createExperiment: fail,
    cancelExperiment: fail,
    getResults: fail,
    listWorkflows: fail,
  };
}
