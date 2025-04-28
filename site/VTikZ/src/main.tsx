import './index.css';
import Home from './Home.tsx';
import LeaderBoard from './LeaderBoard.tsx';
import { HashRouter, Routes, Route } from "react-router";
import ReactDOM from "react-dom/client";

const root = document.getElementById("root")!;

ReactDOM.createRoot(root).render(
  <HashRouter basename="/VTikZ">
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="leaderboard" element={<LeaderBoard />} />
    </Routes>
  </HashRouter>
);