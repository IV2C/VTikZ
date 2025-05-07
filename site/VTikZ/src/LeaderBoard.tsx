import './LeaderBoard.css'
import { BarChart } from '@mui/x-charts/BarChart';
import leaderboard from "./assets/leaderboard.json"
import { Button, Card, CardActions, CardContent, FormControl, InputLabel, MenuItem, Select, SelectChangeEvent, Typography } from '@mui/material';
import React, { useState } from 'react';



function LeaderBoard() {
  const [version, setVersion] = useState("v1.0")
  const handleChangeVersion = (event: SelectChangeEvent) => {
    setVersion(event.target.value as string);
  };
  const rawData = leaderboard as Record<string, {
    model: Record<string, string>,
    total: Record<string, number>,
    Compile: Record<string, number>,
    Location: Record<string, number>,
    SuccessCustomization: Record<string, number>,
    "temp.": Record<string, string>,
    N: Record<string, string>,
    Modality: Record<string, string>,
  }>;
  const data = rawData[version];
  const keys = Object.keys(data.model);

  const all_versions = Object.keys(rawData)

  const chartData = keys.map((id) => ({
    model: data.model[id],
    Compile: data.Compile[id],
    Location: data.Location[id],
    SuccessCustomization: data.SuccessCustomization[id],
    id: data.model[id] + "\nT." + data['temp.'][id] + " " + data.Modality[id] + " N=" + data.N[id],
    temperature: data['temp.'][id],
    modality: data.Modality[id],
    N: data.N[id]
  })).sort((a, b) => b.SuccessCustomization - a.SuccessCustomization);



  return <div className="leaderboard">
    <h1>VTikZ LeaderBoard 👀</h1>


    {/* DropDown menu */}
    <FormControl>
      <InputLabel id="demo-simple-select-label">Version</InputLabel>
      <Select
        labelId="demo-simple-select-label"
        id="demo-simple-select"
        value={version}
        label="Age"
        onChange={handleChangeVersion}
        defaultValue={"v1.0"}
      >
        {all_versions.map(version => (
          <MenuItem value={version}>{version}</MenuItem>))
        }      </Select>
    </FormControl>


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
    <Card className="doc" variant="outlined">
      <React.Fragment>
        <CardContent>
          <Typography variant="h6" component="div">
            Notes
          </Typography>
          <Typography variant="body2">
            <p>Each model name is displayed on the left, and under the name are 3 parameters used in the evaluation:
              <ul>
                <li><b>T</b>: The temperature used in the evaluation</li>
                <li><b>Text/Text+Image</b>: The modalities used, i.e. whether only the text or the text and the image have been provided as input tothe LLM</li>
                <li><b>N</b>: The number of tries given to the LLM to achieve the best score</li>
              </ul>
              The three metrics displayed are : <i>Compile</i>, the LLM creates a code that can be compiled, <i>Location</i>, whether it finds the right line(s) to edit in the code, and <i>SuccessCustomization</i>, whether or not it makes a right customization.
            </p>
          </Typography>
        </CardContent>
        <CardActions>
          <Button href='https://hal.science/hal-05049250' size="small">Learn More</Button>
        </CardActions>
      </React.Fragment>

    </Card>


  </div>
}
export default LeaderBoard