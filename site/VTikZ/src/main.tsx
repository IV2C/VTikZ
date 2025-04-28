import './index.css'
import Home from './Home.tsx'
import LeaderBoard from './LeaderBoard.tsx'
import {
  createBrowserRouter,
  RouterProvider,
} from "react-router";

import ReactDOM from "react-dom/client";

const router = createBrowserRouter(
  [
    {
      path: "/",
      children: [
        { path: "", element: <Home /> },
        { path: "leaderboard", element: <LeaderBoard /> },
      ],
    },
  ],
  {
    basename: "/VTikZ",
  }
);

const root = document.getElementById("root")!;

ReactDOM.createRoot(root).render(
  <RouterProvider router={router} />
);


