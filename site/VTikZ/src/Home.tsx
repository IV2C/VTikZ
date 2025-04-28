import './Home.css'
import context from "./assets/context.jpg"
import { Link } from 'react-router';

function Home() {
  return (

    <div className="paper-page">
      <header className="paper-header">
        <h1>LLM Code Customization with Visual Results: A Benchmark on TikZ</h1>
        <span className="authors">
        <a href='https://scholar.google.com/citations?user=gxLMcIkAAAAJ'>Charly Reux</a>,
        <a href='https://scholar.google.com/citations?user=cC6xt1YAAAAJ'>Mathieu Acher, </a>
        <a href='https://scholar.google.com/citations?user=2Pxbbs0AAAAJ'>Djamel Eddine Khelladi, </a>
        <a href='https://scholar.google.com/citations?user=1nx4T-cAAAAJ'>Clément Quinton, </a>
        <a href='https://scholar.google.fr/citations?user=LV5i8X4AAAAJ'>Olivier Barais</a>

        </span>
        <div className="badges">
          <a href="https://arxiv.com" target="_blank" >
            <span className="badge">📄 Paper</span>
          </a>
          <link href="/VTikZ/leaderboard">
            <span className="badge">📊 LeaderBoard</span>
          </link>
          <a href="https://huggingface.co/datasets/CharlyR/vtikz" target="_blank">
            <span className="badge">🤗 Dataset</span>
          </a>
          <a href="https://github.com/IV2C/VTikZ" target="_blank">
            <span className="badge">💻 Code</span>
          </a>
        </div>
      </header>

      <section className="image-context">
        <img src={context}></img>
      </section>

      <div className='text-sections'>
        <section className="abstract">
          <h2>Abstract</h2>
          <p>With the rise of AI-based code generation, customizing existingcode out of natural language instructions to modify visual results– such as figures or images – has become possible, promising to reduce the need for deep programming expertise. However, evenexperienced developers can struggle with this task, as it requiresidentifying relevant code regions (feature location), generating validcode variants, and ensuring the modifications reliably align withuser intent. In this paper, we introduce vTikZ, the first benchmarkdesigned to evaluate the ability of Large Language Models (LLMs)to customize code while preserving coherent visual outcomes. Ourbenchmark consists of carefully curated vTikZ editing scenarios,parameterized ground truths, and a reviewing tool that leveragesvisual feedback to assess correctness. Empirical evaluation with state-of-the-art LLMs shows that existing solutions struggle to reliablymodify code in alignment with visual intent, highlighting a gap incurrent AI-assisted code editing approaches. We argue that vTikZopens new research directions for integrating LLMs with visualfeedback mechanisms to improve code customization tasks in variousdomains beyond TikZ, including image processing, art creation, Webdesign, and 3D modeling</p>
        </section>

        <section className="bibtex">
          <h2>BibTeX</h2>
          <pre>
            {`@inproceedings{yourBibtexKey,
  author    = {Author Name(s)},
  title     = {LLM Code Customization with Visual Results: A Benchmark on TikZ},
  booktitle = {Conference Name},
  year      = {2025},
}`}
          </pre>
        </section>
      </div>
    </div>
  );
}

export default Home;