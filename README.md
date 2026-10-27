# 🔍 TruthLens AI v2.5

**Instant Amazon Review & Fake Detection Intelligence.** Real-time online scraping, sentiment analysis, authenticity scoring, and competitive product comparison powered by AI.

## ✨ Features

### 🔍 **Core Analytics**
- **Advanced Sentiment Analysis** - Uses TextBlob and spaCy for deep sentiment understanding
- **Authenticity Detection** - AI-powered fake review detection with linguistic pattern analysis
- **Content Categorization** - Automatically categorizes reviews by product aspects (Quality, Performance, Design, etc.)
- **Linguistic Feature Analysis** - POS tags, named entities, and dependency parsing

### 🎨 **Enhanced UI/UX**
- **Dark/Light Mode** - Toggle between themes with persistent preferences
- **Responsive Design** - Mobile-first approach with adaptive layouts
- **Real-time Progress** - Live progress bars and loading states
- **Interactive Visualizations** - Dynamic charts and graphs with Plotly

### ⚡ **Performance & Reliability**
- **Intelligent Caching** - SQLite-based caching system for faster repeated analyses
- **Retry Logic** - Exponential backoff for robust web scraping
- **Input Validation** - Comprehensive URL validation and sanitization
- **Error Handling** - Graceful error recovery with user-friendly messages

### 🔄 **Multi-Product Analysis**
- **Product Comparison** - Compare insights across multiple products
- **Historical Tracking** - Track changes in review patterns over time
- **Batch Processing** - Analyze multiple products simultaneously

### 📊 **Advanced Export Options**
- **Multiple Formats** - CSV, Excel, JSON, and Markdown exports
- **Comprehensive Reports** - Auto-generated analysis reports with insights
- **API Integration** - RESTful API for programmatic access

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Chrome browser (for web scraping)
- 4GB+ RAM recommended

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd amazon-review-insight-pro

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the application
streamlit run app.py
```

### Windows Setup
```bash
# Use the provided setup script
setup.bat

# Or run manually
run.bat
```

## 📁 Project Structure

```
amazon-review-insight-pro/
├── app.py                 # Main Streamlit application
├── utils.py              # Utility functions and configuration
├── scraper.py            # Web scraping functionality
├── nlp_analyzer.py       # NLP analysis components
├── config.yaml           # Application configuration
├── requirements.txt      # Python dependencies
├── setup.bat            # Windows setup script
├── run.bat              # Windows run script
├── README.md            # This file
└── review_cache.db      # SQLite cache database (auto-generated)
```

## 🔧 Configuration

The application uses `config.yaml` for configuration. Key settings:

```yaml
scraping:
  max_reviews_per_page: 50
  retry_attempts: 3
  timeout_seconds: 30
  cache_duration_hours: 24

nlp:
  sentiment_threshold_positive: 0.2
  sentiment_threshold_negative: -0.2
  authenticity_threshold_high: 70
  authenticity_threshold_medium: 40

ui:
  default_theme: "light"
  animations_enabled: true
  responsive_breakpoint: 768
```

## 🚀 Usage

### Basic Analysis
1. **Enter Product URL** - Paste any Amazon product URL
2. **Click Analyze** - The app will scrape and analyze reviews
3. **View Results** - Explore sentiment, authenticity, and insights

### Multi-Product Comparison
1. **Analyze First Product** - Complete initial analysis
2. **Add Comparison** - Enter additional product URLs
3. **Compare Insights** - View side-by-side comparisons

### Export Data
- **CSV/Excel** - Raw data for further analysis
- **JSON** - Structured data for APIs
- **Markdown** - Human-readable reports

## 🔍 How It Works

### 1. **Web Scraping**
- Uses Selenium with Chrome WebDriver
- Anti-detection measures (random user agents, delays)
- Retry logic with exponential backoff
- Intelligent caching to avoid re-scraping

### 2. **NLP Analysis**
- **Sentiment Analysis**: TextBlob + spaCy for polarity and subjectivity
- **Content Categorization**: Keyword-based classification of review topics
- **Authenticity Detection**: Linguistic pattern analysis for fake review detection
- **Feature Extraction**: POS tagging, named entity recognition, dependency parsing

### 3. **Data Processing**
- Structured data extraction from raw text
- Statistical analysis and metrics calculation
- Visualization generation with Plotly
- Report generation with insights and recommendations

## 📊 Analytics Features

### Sentiment Analysis
- **Polarity Score**: -1 (negative) to +1 (positive)
- **Subjectivity Score**: 0 (objective) to 1 (subjective)
- **Sentiment Categories**: Positive, Negative, Neutral

### Authenticity Detection
- **Linguistic Patterns**: First-person pronouns, specific details, emotional language
- **Generic Phrase Detection**: Identifies common fake review phrases
- **Authenticity Score**: 0-100 scale with confidence levels

### Content Categorization
- **Quality**: Material, build quality, durability
- **Performance**: Speed, efficiency, power
- **Design**: Appearance, style, color
- **Value**: Price, cost-effectiveness
- **Features**: Functionality, specifications
- **Usability**: Ease of use, user interface
- **Service**: Delivery, support, warranty

## 🛡️ Security & Privacy

- **Input Sanitization**: XSS protection and input validation
- **Rate Limiting**: Prevents abuse and respects website policies
- **Local Storage**: All data processed locally, no external API calls
- **Secure Scraping**: Respects robots.txt and implements delays

## 🔧 Troubleshooting

### Common Issues

**Chrome Driver Issues**
```bash
# Update Chrome and ChromeDriver
pip install --upgrade webdriver-manager
```

**spaCy Model Missing**
```bash
python -m spacy download en_core_web_sm
```

**Memory Issues**
- Reduce `max_reviews_per_page` in config
- Clear cache: Delete `review_cache.db`

**Scraping Failures**
- Check internet connection
- Verify Amazon URL format
- Try different product (some may have anti-bot protection)

### Performance Optimization

**For Large Datasets**
- Enable caching in config
- Use batch processing for multiple products
- Consider running on a server with more RAM

**For Faster Analysis**
- Reduce retry attempts
- Lower timeout values
- Use cached data when possible

## ☁️ Free Cloud Deployment (Streamlit Community Cloud)

You can host **TruthLens AI** online 100% free with 1 click:
1. Push this repository to your GitHub account (`https://github.com/Hemankumar2501/TruthLens-Ai`).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New app"** and select:
   - **Repository:** `Hemankumar2501/TruthLens-Ai`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy"**! Streamlit Cloud will automatically install dependencies and launch your public URL.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit** - For the amazing web framework
- **spaCy** - For advanced NLP capabilities
- **TextBlob** - For sentiment analysis
- **Selenium** - For web scraping
- **Plotly** - For interactive visualizations

## 📞 Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration options

---

**Made with ❤️ for better e-commerce insights**
