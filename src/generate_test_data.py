"""
generate_test_data.py — Creates synthetic MCP tool registries and test queries.

Generates tool sets at 4 scales (10, 25, 50, 100 tools) with
corresponding test queries and ground truth data.

Design Reference: implementation_poa.md § Q1 (Test Data)
"""

import json
import random
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════
#  TOOL DEFINITIONS — Realistic MCP tool schemas across categories
# ══════════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    # ── Category: Weather & Location ──
    {
        "name": "get_weather",
        "description": "Get current weather conditions for a specified city including temperature, humidity, and conditions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name (e.g., 'Paris', 'Tokyo')"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units"}
            },
            "required": ["city"]
        },
        "annotations": {"category": "weather"}
    },
    {
        "name": "get_forecast",
        "description": "Get a multi-day weather forecast for a location.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {"type": "integer", "description": "Number of days (1-14)"}
            },
            "required": ["city"]
        },
        "annotations": {"category": "weather"}
    },
    {
        "name": "geocode_address",
        "description": "Convert a street address to latitude and longitude coordinates.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "Full street address"}
            },
            "required": ["address"]
        },
        "annotations": {"category": "location"}
    },

    # ── Category: Communication ──
    {
        "name": "send_email",
        "description": "Send an email to a recipient with subject and body content.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject line"},
                "body": {"type": "string", "description": "Email body text"}
            },
            "required": ["to", "subject", "body"]
        },
        "annotations": {"category": "communication"}
    },
    {
        "name": "send_slack_message",
        "description": "Send a message to a Slack channel or user.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "description": "Slack channel name or user ID"},
                "message": {"type": "string", "description": "Message text"}
            },
            "required": ["channel", "message"]
        },
        "annotations": {"category": "communication"}
    },
    {
        "name": "create_calendar_event",
        "description": "Create a new calendar event with title, time, and optional attendees.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title"},
                "start_time": {"type": "string", "description": "Start time in ISO 8601 format"},
                "end_time": {"type": "string", "description": "End time in ISO 8601 format"},
                "attendees": {"type": "array", "items": {"type": "string"}, "description": "List of attendee emails"}
            },
            "required": ["title", "start_time", "end_time"]
        },
        "annotations": {"category": "communication"}
    },
    {
        "name": "send_sms",
        "description": "Send an SMS text message to a phone number.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "phone_number": {"type": "string", "description": "Recipient phone number"},
                "message": {"type": "string", "description": "SMS text content"}
            },
            "required": ["phone_number", "message"]
        },
        "annotations": {"category": "communication"}
    },

    # ── Category: Data & Storage ──
    {
        "name": "read_file",
        "description": "Read the contents of a file at the specified path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"}
            },
            "required": ["path"]
        },
        "annotations": {"category": "filesystem"}
    },
    {
        "name": "write_file",
        "description": "Write content to a file at the specified path. Creates the file if it doesn't exist.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write to"},
                "content": {"type": "string", "description": "Content to write"}
            },
            "required": ["path", "content"]
        },
        "annotations": {"category": "filesystem"}
    },
    {
        "name": "list_directory",
        "description": "List all files and subdirectories in a given directory path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list"}
            },
            "required": ["path"]
        },
        "annotations": {"category": "filesystem"}
    },
    {
        "name": "query_database",
        "description": "Execute a SQL query on the connected PostgreSQL database and return results.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL query to execute"},
                "database": {"type": "string", "description": "Database name"}
            },
            "required": ["query"]
        },
        "annotations": {"category": "database"}
    },
    {
        "name": "list_tables",
        "description": "List all tables in the connected database with their column information.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "database": {"type": "string", "description": "Database name"}
            },
            "required": []
        },
        "annotations": {"category": "database"}
    },

    # ── Category: Web & APIs ──
    {
        "name": "http_get",
        "description": "Make an HTTP GET request to a URL and return the response.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
                "headers": {"type": "object", "description": "Optional HTTP headers"}
            },
            "required": ["url"]
        },
        "annotations": {"category": "web"}
    },
    {
        "name": "http_post",
        "description": "Make an HTTP POST request to a URL with a JSON body.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to post to"},
                "body": {"type": "object", "description": "JSON request body"},
                "headers": {"type": "object", "description": "Optional HTTP headers"}
            },
            "required": ["url", "body"]
        },
        "annotations": {"category": "web"}
    },
    {
        "name": "scrape_webpage",
        "description": "Extract text content from a webpage URL, removing HTML tags.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Webpage URL to scrape"}
            },
            "required": ["url"]
        },
        "annotations": {"category": "web"}
    },
    {
        "name": "search_web",
        "description": "Search the web using a search engine and return top results.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Number of results to return"}
            },
            "required": ["query"]
        },
        "annotations": {"category": "web"}
    },

    # ── Category: ML & Analytics ──
    {
        "name": "analyze_csv",
        "description": "Analyze a CSV file and return summary statistics including mean, median, and correlations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the CSV file"},
                "columns": {"type": "array", "items": {"type": "string"}, "description": "Specific columns to analyze"}
            },
            "required": ["file_path"]
        },
        "annotations": {"category": "analytics"}
    },
    {
        "name": "create_chart",
        "description": "Create a chart or plot from provided data and save as an image.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chart_type": {"type": "string", "enum": ["bar", "line", "pie", "scatter"], "description": "Type of chart"},
                "data": {"type": "object", "description": "Chart data with labels and values"},
                "title": {"type": "string", "description": "Chart title"},
                "output_path": {"type": "string", "description": "Path to save the chart image"}
            },
            "required": ["chart_type", "data"]
        },
        "annotations": {"category": "analytics"}
    },
    {
        "name": "run_inference",
        "description": "Run inference on a machine learning model with given input data.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_name": {"type": "string", "description": "Name of the ML model"},
                "input_data": {"type": "object", "description": "Input data for the model"}
            },
            "required": ["model_name", "input_data"]
        },
        "annotations": {"category": "ml"}
    },
    {
        "name": "get_embeddings",
        "description": "Generate text embeddings using a specified embedding model.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to embed"},
                "model": {"type": "string", "description": "Embedding model name"}
            },
            "required": ["text"]
        },
        "annotations": {"category": "ml"}
    },
    {
        "name": "classify_text",
        "description": "Classify text into predefined categories using NLP.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to classify"},
                "categories": {"type": "array", "items": {"type": "string"}, "description": "List of possible categories"}
            },
            "required": ["text", "categories"]
        },
        "annotations": {"category": "ml"}
    },

    # ── Category: System & Utilities ──
    {
        "name": "get_current_time",
        "description": "Get the current date and time in a specified timezone.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "IANA timezone (e.g., 'America/New_York')"}
            },
            "required": []
        },
        "annotations": {"category": "system"}
    },
    {
        "name": "run_shell_command",
        "description": "Execute a shell command on the system and return its output.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds"}
            },
            "required": ["command"]
        },
        "annotations": {"category": "system"}
    },
    {
        "name": "get_system_info",
        "description": "Get system information including CPU, memory, and disk usage.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        },
        "annotations": {"category": "system"}
    },

    # ── Category: Domain-Specific ──
    {
        "name": "translate_text",
        "description": "Translate text from one language to another.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to translate"},
                "source_language": {"type": "string", "description": "Source language code (e.g., 'en')"},
                "target_language": {"type": "string", "description": "Target language code (e.g., 'fr')"}
            },
            "required": ["text", "target_language"]
        },
        "annotations": {"category": "text"}
    },
    {
        "name": "summarize_text",
        "description": "Generate a concise summary of a long text document.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to summarize"},
                "max_length": {"type": "integer", "description": "Maximum summary length in words"}
            },
            "required": ["text"]
        },
        "annotations": {"category": "text"}
    },
    {
        "name": "generate_image",
        "description": "Generate an image from a text description using AI.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Text description of the image to generate"},
                "width": {"type": "integer", "description": "Image width in pixels"},
                "height": {"type": "integer", "description": "Image height in pixels"}
            },
            "required": ["prompt"]
        },
        "annotations": {"category": "media"}
    },
    {
        "name": "get_stock_price",
        "description": "Get the current or historical stock price for a given ticker symbol.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol (e.g., 'AAPL')"},
                "date": {"type": "string", "description": "Optional date for historical price (YYYY-MM-DD)"}
            },
            "required": ["symbol"]
        },
        "annotations": {"category": "finance"}
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another using current exchange rates.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to convert"},
                "from_currency": {"type": "string", "description": "Source currency code (e.g., 'USD')"},
                "to_currency": {"type": "string", "description": "Target currency code (e.g., 'EUR')"}
            },
            "required": ["amount", "from_currency", "to_currency"]
        },
        "annotations": {"category": "finance"}
    },

    # ── Category: Git & Code ──
    {
        "name": "git_clone",
        "description": "Clone a Git repository to a local directory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_url": {"type": "string", "description": "Git repository URL"},
                "destination": {"type": "string", "description": "Local directory path"}
            },
            "required": ["repo_url"]
        },
        "annotations": {"category": "git"}
    },
    {
        "name": "git_commit",
        "description": "Create a Git commit with a message in the current repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Commit message"},
                "files": {"type": "array", "items": {"type": "string"}, "description": "Files to stage"}
            },
            "required": ["message"]
        },
        "annotations": {"category": "git"}
    },
    {
        "name": "create_github_issue",
        "description": "Create a new issue on a GitHub repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository in 'owner/repo' format"},
                "title": {"type": "string", "description": "Issue title"},
                "body": {"type": "string", "description": "Issue description"}
            },
            "required": ["repo", "title"]
        },
        "annotations": {"category": "git"}
    },
    {
        "name": "search_code",
        "description": "Search through code files for a pattern or string using regex.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern (regex supported)"},
                "directory": {"type": "string", "description": "Directory to search in"},
                "file_type": {"type": "string", "description": "File extension filter (e.g., '.py')"}
            },
            "required": ["pattern"]
        },
        "annotations": {"category": "code"}
    },

    # ── Category: Knowledge & Search ──
    {
        "name": "search_wikipedia",
        "description": "Search Wikipedia for articles matching a query and return summaries.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "language": {"type": "string", "description": "Wikipedia language code (default: 'en')"}
            },
            "required": ["query"]
        },
        "annotations": {"category": "knowledge"}
    },
    {
        "name": "lookup_definition",
        "description": "Look up the dictionary definition of a word.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "word": {"type": "string", "description": "Word to look up"}
            },
            "required": ["word"]
        },
        "annotations": {"category": "knowledge"}
    },

    # ── Category: Task Management ──
    {
        "name": "create_task",
        "description": "Create a new task or to-do item with a title and optional due date.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Task title"},
                "due_date": {"type": "string", "description": "Due date in YYYY-MM-DD format"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"], "description": "Task priority"}
            },
            "required": ["title"]
        },
        "annotations": {"category": "productivity"}
    },
    {
        "name": "list_tasks",
        "description": "List all tasks with optional filtering by status or priority.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["pending", "completed", "all"], "description": "Filter by status"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"], "description": "Filter by priority"}
            },
            "required": []
        },
        "annotations": {"category": "productivity"}
    },

    # ── Category: Security & Auth ──
    {
        "name": "generate_password",
        "description": "Generate a secure random password with specified requirements.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "length": {"type": "integer", "description": "Password length"},
                "include_symbols": {"type": "boolean", "description": "Include special characters"},
                "include_numbers": {"type": "boolean", "description": "Include numbers"}
            },
            "required": ["length"]
        },
        "annotations": {"category": "security"}
    },
    {
        "name": "hash_text",
        "description": "Generate a cryptographic hash of input text using a specified algorithm.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to hash"},
                "algorithm": {"type": "string", "enum": ["sha256", "sha512", "md5"], "description": "Hash algorithm"}
            },
            "required": ["text"]
        },
        "annotations": {"category": "security"}
    },

    # ── Additional filler tools to reach 50 ──
    {
        "name": "resize_image",
        "description": "Resize an image to specified dimensions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "description": "Input image path"},
                "width": {"type": "integer", "description": "Target width"},
                "height": {"type": "integer", "description": "Target height"},
                "output_path": {"type": "string", "description": "Output image path"}
            },
            "required": ["input_path", "width", "height"]
        },
        "annotations": {"category": "media"}
    },
    {
        "name": "convert_pdf_to_text",
        "description": "Extract text content from a PDF file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pdf_path": {"type": "string", "description": "Path to the PDF file"}
            },
            "required": ["pdf_path"]
        },
        "annotations": {"category": "document"}
    },
    {
        "name": "compress_files",
        "description": "Compress files or directories into a zip archive.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Files/directories to compress"},
                "output_path": {"type": "string", "description": "Output zip file path"}
            },
            "required": ["paths", "output_path"]
        },
        "annotations": {"category": "filesystem"}
    },
    {
        "name": "validate_json",
        "description": "Validate JSON data against a JSON schema.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "data": {"type": "object", "description": "JSON data to validate"},
                "schema": {"type": "object", "description": "JSON schema to validate against"}
            },
            "required": ["data", "schema"]
        },
        "annotations": {"category": "data"}
    },
    {
        "name": "parse_csv",
        "description": "Parse a CSV file and return structured data as JSON.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to CSV file"},
                "delimiter": {"type": "string", "description": "Column delimiter character"}
            },
            "required": ["file_path"]
        },
        "annotations": {"category": "data"}
    },
    {
        "name": "create_spreadsheet",
        "description": "Create an Excel spreadsheet with specified data and formatting.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "description": "Rows of data"},
                "headers": {"type": "array", "items": {"type": "string"}, "description": "Column headers"},
                "output_path": {"type": "string", "description": "Output file path"}
            },
            "required": ["data", "output_path"]
        },
        "annotations": {"category": "document"}
    },
    {
        "name": "ocr_image",
        "description": "Extract text from an image using optical character recognition.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Path to image file"},
                "language": {"type": "string", "description": "OCR language (default: 'eng')"}
            },
            "required": ["image_path"]
        },
        "annotations": {"category": "media"}
    },
    {
        "name": "calculate_math",
        "description": "Evaluate a mathematical expression and return the result.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
            },
            "required": ["expression"]
        },
        "annotations": {"category": "utility"}
    },
    {
        "name": "format_date",
        "description": "Convert a date string from one format to another.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "date_string": {"type": "string", "description": "Input date string"},
                "input_format": {"type": "string", "description": "Input format pattern"},
                "output_format": {"type": "string", "description": "Desired output format pattern"}
            },
            "required": ["date_string", "output_format"]
        },
        "annotations": {"category": "utility"}
    },
    {
        "name": "generate_uuid",
        "description": "Generate a universally unique identifier (UUID).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "version": {"type": "integer", "enum": [1, 4], "description": "UUID version"}
            },
            "required": []
        },
        "annotations": {"category": "utility"}
    },
]


# ══════════════════════════════════════════════════════════════════════
#  TEST QUERIES + GROUND TRUTH
# ══════════════════════════════════════════════════════════════════════

TEST_QUERIES = [
    # ── Simple selection (1 obvious tool) ──
    {"query": "What's the weather like in London right now?", "expected_tool": "get_weather", "expected_params": {"city": "London"}, "category": "simple", "difficulty": "easy"},
    {"query": "Send an email to alice@company.com with subject 'Meeting tomorrow' and body 'See you at 3pm'", "expected_tool": "send_email", "expected_params": {"to": "alice@company.com", "subject": "Meeting tomorrow"}, "category": "simple", "difficulty": "easy"},
    {"query": "Read the contents of /home/user/notes.txt", "expected_tool": "read_file", "expected_params": {"path": "/home/user/notes.txt"}, "category": "simple", "difficulty": "easy"},
    {"query": "What's Apple's current stock price?", "expected_tool": "get_stock_price", "expected_params": {"symbol": "AAPL"}, "category": "simple", "difficulty": "easy"},
    {"query": "Translate 'Hello world' to French", "expected_tool": "translate_text", "expected_params": {"text": "Hello world", "target_language": "fr"}, "category": "simple", "difficulty": "easy"},
    {"query": "What time is it in New York?", "expected_tool": "get_current_time", "expected_params": {"timezone": "America/New_York"}, "category": "simple", "difficulty": "easy"},
    {"query": "Generate a 16-character secure password", "expected_tool": "generate_password", "expected_params": {"length": 16}, "category": "simple", "difficulty": "easy"},
    {"query": "Search the web for 'best python frameworks 2025'", "expected_tool": "search_web", "expected_params": {"query": "best python frameworks 2025"}, "category": "simple", "difficulty": "easy"},
    {"query": "List all files in the /var/log directory", "expected_tool": "list_directory", "expected_params": {"path": "/var/log"}, "category": "simple", "difficulty": "easy"},
    {"query": "Create a bar chart showing sales data", "expected_tool": "create_chart", "expected_params": {"chart_type": "bar"}, "category": "simple", "difficulty": "easy"},

    # ── Moderate difficulty (requires disambiguation) ──
    {"query": "I need to find information about quantum computing", "expected_tool": "search_wikipedia", "expected_params": {"query": "quantum computing"}, "category": "simple", "difficulty": "medium"},
    {"query": "Clone the repository at https://github.com/user/project", "expected_tool": "git_clone", "expected_params": {"repo_url": "https://github.com/user/project"}, "category": "simple", "difficulty": "medium"},
    {"query": "Post a message in the #engineering Slack channel saying 'Build passed!'", "expected_tool": "send_slack_message", "expected_params": {"channel": "#engineering", "message": "Build passed!"}, "category": "simple", "difficulty": "medium"},
    {"query": "What does the word 'ephemeral' mean?", "expected_tool": "lookup_definition", "expected_params": {"word": "ephemeral"}, "category": "simple", "difficulty": "medium"},
    {"query": "Run SELECT * FROM users WHERE active = true on the database", "expected_tool": "query_database", "expected_params": {"query": "SELECT * FROM users WHERE active = true"}, "category": "simple", "difficulty": "medium"},
    {"query": "Extract text from the scanned document at /docs/receipt.jpg", "expected_tool": "ocr_image", "expected_params": {"image_path": "/docs/receipt.jpg"}, "category": "simple", "difficulty": "medium"},
    {"query": "Convert 100 USD to EUR", "expected_tool": "convert_currency", "expected_params": {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}, "category": "simple", "difficulty": "medium"},
    {"query": "Summarize this long article for me", "expected_tool": "summarize_text", "expected_params": {}, "category": "simple", "difficulty": "medium"},
    {"query": "Create a high-priority task called 'Review PR #42' due tomorrow", "expected_tool": "create_task", "expected_params": {"title": "Review PR #42", "priority": "high"}, "category": "simple", "difficulty": "medium"},
    {"query": "Get the SHA-256 hash of the text 'hello123'", "expected_tool": "hash_text", "expected_params": {"text": "hello123", "algorithm": "sha256"}, "category": "simple", "difficulty": "medium"},

    # ── Hard (ambiguous or requires inference) ──
    {"query": "I want to save some data as a zip file for backup", "expected_tool": "compress_files", "expected_params": {}, "category": "simple", "difficulty": "hard"},
    {"query": "Set up a team meeting for next Monday at 2pm", "expected_tool": "create_calendar_event", "expected_params": {}, "category": "simple", "difficulty": "hard"},
    {"query": "I need to check if this JSON is valid against our API schema", "expected_tool": "validate_json", "expected_params": {}, "category": "simple", "difficulty": "hard"},
    {"query": "What's the 5-day forecast for San Francisco?", "expected_tool": "get_forecast", "expected_params": {"city": "San Francisco", "days": 5}, "category": "simple", "difficulty": "hard"},
    {"query": "Find all Python files in /src that contain the word 'TODO'", "expected_tool": "search_code", "expected_params": {"pattern": "TODO", "file_type": ".py"}, "category": "simple", "difficulty": "hard"},

    # ── Discovery queries (no single correct tool) ──
    {"query": "What tools do I have for working with files?", "expected_tool": "DISCOVERY", "expected_params": {}, "category": "discovery", "difficulty": "easy"},
    {"query": "Show me the communication-related tools available", "expected_tool": "DISCOVERY", "expected_params": {}, "category": "discovery", "difficulty": "easy"},
    {"query": "What machine learning capabilities are available?", "expected_tool": "DISCOVERY", "expected_params": {}, "category": "discovery", "difficulty": "medium"},
]


def generate_test_data(output_dir: str = "."):
    """Generate all test data files."""
    base = Path(output_dir)

    # ── Tool Registries ──
    registries_dir = base / "test_data" / "tool_registries"
    registries_dir.mkdir(parents=True, exist_ok=True)

    sizes = {"small_10": 10, "medium_25": 25, "large_50": 50}
    for name, count in sizes.items():
        tools = ALL_TOOLS[:count]
        with open(registries_dir / f"{name}.json", "w") as f:
            json.dump({"tools": tools, "count": len(tools)}, f, indent=2)
        print(f"  Created {name}.json ({len(tools)} tools)")

    # full set = all tools
    with open(registries_dir / "xlarge_100.json", "w") as f:
        json.dump({"tools": ALL_TOOLS, "count": len(ALL_TOOLS)}, f, indent=2)
    print(f"  Created xlarge_100.json ({len(ALL_TOOLS)} tools)")

    # ── Queries ──
    queries_dir = base / "test_data" / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)

    queries_out = [
        {"query": q["query"], "category": q["category"], "difficulty": q["difficulty"]}
        for q in TEST_QUERIES
    ]
    with open(queries_dir / "all_queries.json", "w") as f:
        json.dump(queries_out, f, indent=2)
    print(f"  Created all_queries.json ({len(queries_out)} queries)")

    # Deterministic train/dev/test split manifests for leakage-safe experiments
    rng = random.Random(42)
    indices = list(range(len(TEST_QUERIES)))

    # Keep category mix stable (simple vs discovery)
    simple_idx = [i for i in indices if TEST_QUERIES[i]["category"] == "simple"]
    discovery_idx = [i for i in indices if TEST_QUERIES[i]["category"] == "discovery"]
    rng.shuffle(simple_idx)
    rng.shuffle(discovery_idx)

    # 70/15/15 on simple queries + deterministic split for discovery (2/0/1)
    n_simple = len(simple_idx)
    n_train_simple = int(round(n_simple * 0.70))
    n_dev_simple = int(round(n_simple * 0.15))
    n_test_simple = n_simple - n_train_simple - n_dev_simple

    train_idx = simple_idx[:n_train_simple] + discovery_idx[:2]
    dev_idx = simple_idx[n_train_simple:n_train_simple + n_dev_simple]
    test_idx = simple_idx[n_train_simple + n_dev_simple:n_train_simple + n_dev_simple + n_test_simple] + discovery_idx[2:]

    split_map = {
        "train": sorted(train_idx),
        "dev": sorted(dev_idx),
        "test": sorted(test_idx),
    }

    for split_name, split_indices in split_map.items():
        split_queries = [
            {
                "query": TEST_QUERIES[i]["query"],
                "category": TEST_QUERIES[i]["category"],
                "difficulty": TEST_QUERIES[i]["difficulty"],
            }
            for i in split_indices
        ]
        with open(queries_dir / f"{split_name}_queries.json", "w") as f:
            json.dump(split_queries, f, indent=2)
        print(f"  Created {split_name}_queries.json ({len(split_queries)} queries)")

    # ── Ground Truth ──
    gt_dir = base / "test_data" / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    gt_out = [
        {"query": q["query"], "expected_tool": q["expected_tool"], "expected_params": q["expected_params"]}
        for q in TEST_QUERIES
    ]
    with open(gt_dir / "expected_selections.json", "w") as f:
        json.dump(gt_out, f, indent=2)
    print(f"  Created expected_selections.json ({len(gt_out)} entries)")

    for split_name, split_indices in split_map.items():
        split_gt = [
            {
                "query": TEST_QUERIES[i]["query"],
                "expected_tool": TEST_QUERIES[i]["expected_tool"],
                "expected_params": TEST_QUERIES[i]["expected_params"],
            }
            for i in split_indices
        ]
        with open(gt_dir / f"{split_name}_expected_selections.json", "w") as f:
            json.dump(split_gt, f, indent=2)
        print(f"  Created {split_name}_expected_selections.json ({len(split_gt)} entries)")

    print(f"\n✅ Test data generation complete! ({len(ALL_TOOLS)} tools, {len(TEST_QUERIES)} queries)")


if __name__ == "__main__":
    generate_test_data()
