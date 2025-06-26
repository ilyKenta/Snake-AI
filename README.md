# Snake AI - Pathfinding Algorithms

A beautiful web application showcasing different pathfinding algorithms in action through an interactive Snake game. Currently features A* pathfinding with plans to add more algorithms.

## 🎮 Live Demo

The application is automatically deployed to GitHub Pages and can be accessed at:
**https://[your-username].github.io/Snake-AI/**

## 🚀 Features

- **Interactive Snake Game**: Watch A* algorithm navigate in real-time
- **Beautiful UI**: Modern, responsive design with algorithm explanations
- **Educational Content**: Detailed breakdown of how A* pathfinding works
- **Extensible Architecture**: Ready for additional algorithms
- **Mobile Friendly**: Works on all devices

## 🧠 Current Algorithms

### A* Pathfinding
- **Heuristic**: Manhattan distance to the fruit
- **Optimality**: Guarantees shortest path when heuristic is admissible
- **Efficiency**: More efficient than Dijkstra's for most cases
- **Formula**: f(n) = g(n) + h(n)

## 🔮 Future Algorithms

- **Dijkstra's Algorithm** - Classic shortest path without heuristics
- **Random Walk** - Simple random movement for comparison
- **Minimax Algorithm** - Game tree search for strategic planning
- **Genetic Algorithm** - Evolutionary approach to pathfinding
- **Neural Network** - Machine learning approach
- **Q-Learning** - Reinforcement learning for optimal policy

## 🛠️ Local Development

### Prerequisites
- Python 3.12+
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/[your-username]/Snake-AI.git
   cd Snake-AI
   ```

2. Install dependencies:
   ```bash
   cd Snake_A_star
   pip install -r requirements.txt
   ```

3. Run locally:
   ```bash
   pygbag .
   ```

4. Open your browser to `http://localhost:8000`

## 🚀 Deployment

This project uses GitHub Actions to automatically deploy to GitHub Pages. The deployment process:

1. **Automatic Trigger**: Deploys on every push to `main` or `master` branch
2. **Build Process**: 
   - Installs Python 3.12
   - Installs pygbag
   - Builds the web application
   - Uploads to GitHub Pages
3. **Live URL**: Available at `https://[your-username].github.io/Snake-AI/`

### Manual Deployment
You can also trigger deployment manually:
1. Go to the "Actions" tab in your GitHub repository
2. Select "Deploy Snake AI to GitHub Pages"
3. Click "Run workflow"

## 📁 Project Structure

```
Snake-AI/
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions deployment workflow
├── Snake_A_star/
│   ├── main.py                 # Main Snake game with A* algorithm
│   ├── requirements.txt        # Python dependencies
│   └── build/
│       └── web/
│           ├── index.html      # Beautiful web interface
│           ├── snake_a_star.apk # Pygbag application bundle
│           └── favicon.png     # Application icon
└── README.md                   # This file
```

## 🔧 Technical Stack

- **Backend**: Python with Pygame
- **Web Framework**: pygbag
- **Frontend**: HTML5, CSS3, JavaScript
- **Styling**: Modern CSS with gradients and animations
- **Icons**: Font Awesome
- **Fonts**: Inter (Google Fonts)
- **Deployment**: GitHub Actions + GitHub Pages

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Adding New Algorithms

To add a new algorithm:

1. Create a new Python file (e.g., `dijkstra_snake.py`)
2. Implement the algorithm following the same interface as the A* version
3. Update the HTML to include a new game frame
4. Add algorithm explanation and features
5. Submit a pull request

## 🙏 Acknowledgments

- Pygame community for the excellent game development framework
- pygbag team for making pygame web-compatible
- Font Awesome for the beautiful icons
- Google Fonts for the Inter font family
- Geeksforgeeks for snake game logic
