```tikz
--- 
\reflect[split={BlueGrey800 and BlueGrey900}]{
  \fill [fill color](0,-144) 
    .. controls ++(  0:20) and ++(315:20) .. ++( 40,64)
    .. controls ++(135:20) and ++( 45:20) .. ++(-80, 0)
    .. controls ++(225:20) and ++(180:20) .. cycle;
  \fill [BlueGrey900] (56, 0) circle [radius=20];
  \fill [fill color] (-8,-112)
  -- ++(16,0) -- ++(0,-32) arc (180:360:24)
  arc (180:0:8) arc (360:180:40);
}
+++ 
\reflect[split={BlueGrey800 and BlueGrey900}]{
  \fill [fill color](0,-144) 
    .. controls ++(  0:20) and ++(315:20) .. ++( 40,64)
    .. controls ++(135:20) and ++( 45:20) .. ++(-80, 0)
    .. controls ++(225:20) and ++(180:20) .. cycle;
  \fill [White] (56, 0) circle [radius=24];
  \fill [BlueGrey900] (56, 0) circle [radius=20];
  \fill [fill color] (-8,-112)
  -- ++(16,0) -- ++(0,-32) arc (180:360:24)
  arc (180:0:8) arc (360:180:40);
}
```