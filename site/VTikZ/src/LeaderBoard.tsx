import './LeaderBoard.css'
import { BarChart } from '@mui/x-charts/BarChart';
import leaderboard from "./assets/leaderboard.json"



function LeaderBoard() {
  const rawData = leaderboard as {
    model: Record<string, string>,
    total: Record<string, number>,
    Compile: Record<string, number>,
    Location: Record<string, number>,
    SuccessCustomization: Record<string, number>,
    "temp.": Record<string, string>,
    N: Record<string, string>,
    Modality: Record<string, string>,
  };
  const keys = Object.keys(rawData.model);
  const chartData = keys.map((id) => ({
    model: rawData.model[id],
    Compile: rawData.Compile[id],
    Location: rawData.Location[id],
    SuccessCustomization: rawData.SuccessCustomization[id],
    id: rawData.model[id] + "\nT." + rawData['temp.'][id] + " " + rawData.Modality[id] + " N=" + rawData.N[id],
    temperature: rawData['temp.'][id],
    modality: rawData.Modality[id],
    N: rawData.N[id]
  })).sort((a, b) => {
    return b.SuccessCustomization - a.SuccessCustomization
  });



  return <div className="leaderboard">
    <h1>VTikZ LeaderBoard 👀</h1>
    <div className='board'>
      <BarChart
        dataset={chartData}
        yAxis={[{
          scaleType: 'band',
          dataKey: "id", label: "model",
          labelStyle: { fontSize: 14 },
          width: 300
        }]} xAxis={[{ scaleType: 'linear', max: 100 }]}
        series={[
          { dataKey: 'Compile', label: 'Compile' },
          { dataKey: 'Location', label: 'Location' },
          { dataKey: 'SuccessCustomization', label: 'SuccessCustomization' },
        ]}
        layout="horizontal"
        grid={{ vertical: true }}
        height={70 * keys.length}
      />

    </div>
  </div>
}
export default LeaderBoard