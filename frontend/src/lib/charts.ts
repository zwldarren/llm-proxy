import {
  Chart as ChartJS,
  Title,
  Tooltip,
  Legend,
  BarElement,
  CategoryScale,
  LinearScale,
} from "chart.js";

let registered = false;

export function registerBarChart(): void {
  if (registered) return;
  ChartJS.register(Title, Tooltip, Legend, BarElement, CategoryScale, LinearScale);
  registered = true;
}
