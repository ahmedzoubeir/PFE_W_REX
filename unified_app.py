from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import os
import tempfile
import traceback
import asyncio
import pandas as pd
import pymysql
from sqlalchemy import create_engine, text, inspect
from werkzeug.utils import secure_filename
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

# Import functions from RAG project
from rag_functions import (UPLOAD_DIR, CHROMA_DIR, allowed_file, find_similar_file, 
                          process_file_for_rag, query_with_rag)

# Configure app
app = Flask(__name__)
app.secret_key = "unified_app_secret_key_123"  # Used for flash messages and sessions

# Create necessary directories
UPLOADS_FOLDER = 'uploads'
os.makedirs(UPLOADS_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# Initialize LLM for SQL project
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    api_key="AIzaSyAVE25hZaDfQz7p8oQxPI--wVpQVxsd_Ak",
    max_tokens=None,
    timeout=None
)

# QA Chain setup for SQL project
system = SystemMessagePromptTemplate.from_template(
    """You are a helpful AI assistant specialized in database queries. You excel at:
    1. Using REGEX instead of LIKE for flexible text matching
    2. Finding similar terms across different languages (e.g., 'cosmetic' in English and 'cosmetique' in French)
    3. Handling spelling variations and typos in search terms
    4. Multilingual support with accented characters
    Always return full results with SELECT * unless user asks otherwise. Never truncate with LIMIT unless instructed.
    Always prioritize finding relevant matches even when terms are not exact."""
)

prompt = """Answer user question based on the provided context ONLY! If you do not know the answer, just say "I don't know".
            ### Context:
            {context}

            ### Question:
            {question}

            ### Answer:"""

prompt = HumanMessagePromptTemplate.from_template(prompt)
messages = [system, prompt]
template = ChatPromptTemplate(messages=messages)
qna_chain = template | llm | StrOutputParser()

# REX Chain setup - New addition for Results EXplainer
rex_system = SystemMessagePromptTemplate.from_template(
    """You are a specialized Results EXplainer (REX) AI assistant. Your role is to:
    1. Analyze SQL query results and provide clear, structured explanations
    2. Summarize key findings from the data
    3. Identify patterns, trends, and important insights
    4. Present information in a clear, actionable format
    5. Handle multilingual content (French, English, etc.)
    6. Provide context-aware explanations based on the original question
    
    Always provide clear, concise explanations that help users understand what the data means."""
)

rex_prompt = """Based on the original question and the SQL query results, please analyze and explain the findings:

### Original Question:
{question}

### SQL Query Results:
{results}

### Number of Records Found:
{count}

Please provide:
1. A summary of what was found
2. Key insights from the data
3. Any patterns or trends observed
4. Actionable recommendations if applicable

### Analysis:"""

rex_prompt_template = HumanMessagePromptTemplate.from_template(rex_prompt)
rex_messages = [rex_system, rex_prompt_template]
rex_template = ChatPromptTemplate(rex_messages)
rex_chain = rex_template | llm | StrOutputParser()

# SQL Project functions
def ask_llm(context, question):
    return qna_chain.invoke({'context': context, 'question': question})

def analyze_results_with_rex(question, results, count):
    """Use REX to analyze and explain SQL query results"""
    try:
        # Format results for better readability
        formatted_results = ""
        if isinstance(results, str):
            # If results is already a string, use it directly
            formatted_results = results
        else:
            # If results is a list of tuples, format it
            for i, row in enumerate(results, 1):
                formatted_results += f"Row {i}: {row}\n"
        
        analysis = rex_chain.invoke({
            'question': question,
            'results': formatted_results,
            'count': count
        })
        
        return analysis
    except Exception as e:
        return f"Error during REX analysis: {str(e)}"

@chain
def get_correct_sql_query(input):
    context = input['context']
    question = input['question']

    instruction = """
        Use the above context to generate a NEW SQL query that answers the following question:
        {}

        IMPORTANT REQUIREMENTS:
        1. Always use REGEXP instead of LIKE for text searching to enable more flexible matching
        2. When searching for terms, implement aggressive pattern matching with REGEXP
        3. For text searches, use patterns that can match across languages
        4. CRITICAL: Double-check your SQL syntax! Ensure all parentheses are properly balanced
        5. DO NOT use complex REGEXP patterns with nested parentheses - keep patterns simple
        6. For case-insensitive matching, use the '(?i)' prefix in your pattern
        7. IMPORTANT: Use SELECT * to return all columns unless the user explicitly asks for specific fields.
        8. DO NOT use LIMIT unless the question includes "show a few rows" or "limit the result".

        The following are JUST EXAMPLES of how to structure queries - DO NOT COPY THESE EXACTLY:
        
        Example 1: For searching cosmetic/aesthetic terms across multiple columns:
        ```
        SELECT * FROM some_table 
        WHERE col1 REGEXP '(?i)cosmet|esthet' 
        OR col2 REGEXP '(?i)cosmet|esthet';
        ```
        
        Example 2: For counting records with specific patterns:
        ```
        SELECT COUNT(*) FROM another_table WHERE description REGEXP '(?i)pattern';
        ```

        CREATE A NEW QUERY tailored specifically to the question above.
        Do not enclose your response in ```sql tags and do not write explanations.
        Return ONLY a single valid SQL query.
    """.format(question)

    response = ask_llm(context=context, question=instruction)
    return response

def get_available_databases():
    """Get list of available MySQL databases"""
    engine = create_engine('mysql+pymysql://root:root@localhost/')
    with engine.connect() as conn:
        result = conn.execute(text("SHOW DATABASES"))
        databases = [row[0] for row in result if row[0] not in ['information_schema', 'mysql', 'performance_schema', 'sys']]
    return databases

def get_database_sample(db_name, limit=10):
    """Get a sample of data from the selected database"""
    engine = create_engine(f'mysql+pymysql://root:root@localhost/{db_name}')
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    if not tables:
        return {"error": "No tables found in database"}, None
    
    # Get first table as sample
    table_name = tables[0]
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT {limit}"))
        columns = result.keys()
        rows = [dict(zip(columns, row)) for row in result]
    
    return rows, table_name

def sanitize_sql(sql_query):
    """Remove markdown code blocks and other formatting from SQL queries"""
    # Remove markdown code block delimiters
    sql_query = sql_query.replace('```sql', '')
    sql_query = sql_query.replace('```', '')
    return sql_query.strip()

def upload_excel_to_mysql(file_path, db_name, table_name=None):
    """Upload Excel file to MySQL database"""
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # If table name not provided, use the filename
    if not table_name:
        table_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Create database if it doesn't exist
    engine = create_engine('mysql+pymysql://root:root@localhost/')
    with engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {db_name}"))
        conn.commit()
    
    # Connect to the database and upload data
    engine = create_engine(f'mysql+pymysql://root:root@localhost/{db_name}')
    df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    
    return {"success": True, "rows": len(df), "columns": len(df.columns)}

def parse_sql_results(results_string):
    """Parse SQL results string into structured data for table display"""
    try:
        # Handle the case where results is already a list
        if isinstance(results_string, list):
            return results_string
        
        # Handle string representation of tuples
        if isinstance(results_string, str):
            import re
            import ast
            
            # Try to evaluate if it's a proper Python list/tuple format
            try:
                # Clean up the string and try to parse as Python literal
                cleaned = results_string.strip()
                if cleaned.startswith('[') and cleaned.endswith(']'):
                    # Try to safely evaluate the list
                    evaluated = ast.literal_eval(cleaned)
                    if isinstance(evaluated, list):
                        return evaluated
            except:
                pass
            
            # Parse tuple patterns manually
            parsed_results = []
            
            # Find all tuples in the string using regex
            tuple_pattern = r"\('([^']*)'(?:,\s*'([^']*)')*\)"
            matches = re.findall(tuple_pattern, results_string)
            
            if matches:
                for match in matches:
                    # Convert match to list, filtering out empty strings
                    row = [item for item in match if item]
                    if row:
                        parsed_results.append(row)
            else:
                # Try alternative parsing for complex tuples
                # Split by '), (' to separate tuples
                if '(' in results_string and ')' in results_string:
                    # Remove outer brackets if present
                    clean_str = results_string.strip()
                    if clean_str.startswith('['):
                        clean_str = clean_str[1:]
                    if clean_str.endswith(']'):
                        clean_str = clean_str[:-1]
                    
                    # Split by tuple boundaries
                    tuple_parts = re.split(r'\),\s*\(', clean_str)
                    
                    for part in tuple_parts:
                        # Clean up the part
                        part = part.strip()
                        if part.startswith('('):
                            part = part[1:]
                        if part.endswith(')'):
                            part = part[:-1]
                        
                        # Split by commas and clean quotes
                        if part:
                            items = []
                            # Use a more sophisticated split that handles quoted commas
                            current_item = ""
                            in_quotes = False
                            quote_char = None
                            
                            i = 0
                            while i < len(part):
                                char = part[i]
                                if char in ["'", '"'] and not in_quotes:
                                    in_quotes = True
                                    quote_char = char
                                elif char == quote_char and in_quotes:
                                    in_quotes = False
                                    quote_char = None
                                elif char == ',' and not in_quotes:
                                    items.append(current_item.strip().strip("'\""))
                                    current_item = ""
                                    i += 1
                                    continue
                                
                                current_item += char
                                i += 1
                            
                            # Add the last item
                            if current_item:
                                items.append(current_item.strip().strip("'\""))
                            
                            if items:
                                parsed_results.append(items)
                else:
                    # Fallback: split by lines
                    lines = results_string.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line:
                            parsed_results.append([line])
            
            return parsed_results
        
        return [[str(results_string)]]  # Fallback for other types
        
    except Exception as e:
        print(f"Error parsing SQL results: {e}")
        # If all parsing fails, return raw string split by lines
        lines = str(results_string).split('\n')
        return [[line.strip()] for line in lines if line.strip()]

# Routes for the unified app
@app.route('/')
def home():
    """Main landing page to choose between RAG and SQL projects"""
    return render_template('home.html')

# RAG Project Routes
@app.route('/rag')
def rag_index():
    """Home page for RAG project with upload form"""
    return render_template('index2.html')

@app.route('/rag/upload', methods=['POST'])
def rag_upload_file():
    """Handle file uploads for RAG project"""
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(url_for('rag_index'))
            
        file = request.files['file']
        
        # Check if user submitted an empty form
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('rag_index'))
            
        # Process file if it's allowed
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_DIR, filename)
            
            # Save the uploaded file
            file.save(file_path)
            
            # Check if similar file exists
            is_duplicate, collection_name = find_similar_file(file_path)
            
            # Select embedding model from form
            model_embed = request.form.get('model_embed', 'nomic-embed-text:latest')
            
            if is_duplicate:
                flash(f'Similar document already exists. Using existing collection: {collection_name}', 'info')
                # Store collection name in session for querying
                session['collection_name'] = collection_name
                return redirect(url_for('rag_query_page'))
            else:
                # Process the file for RAG
                collection_name, is_new = process_file_for_rag(file_path, model_embed)
                
                if is_new:
                    flash(f'Document processed successfully. Created collection: {collection_name}', 'success')
                else:
                    flash(f'Document processed successfully. Using collection: {collection_name}', 'success')
                
                # Store collection name in session for querying
                session['collection_name'] = collection_name
                return redirect(url_for('rag_query_page'))
        else:
            flash(f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}', 'error')
            return redirect(url_for('rag_index'))
            
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        print(traceback.format_exc())
        return redirect(url_for('rag_index'))

@app.route('/rag/query')
def rag_query_page():
    """Render the query page for RAG project"""
    collection_name = session.get('collection_name', None)
    if not collection_name:
        flash('Please upload a document first', 'error')
        return redirect(url_for('rag_index'))
        
    return render_template('query2.html', collection_name=collection_name)

@app.route('/rag/api/query', methods=['POST'])
def rag_api_query():
    """API endpoint for RAG document queries"""
    try:
        data = request.json
        query_text = data.get('query', '')
        collection_name = data.get('collection_name', '')
        model_embed = data.get('model_embed', 'nomic-embed-text:latest')
        model_llm = data.get('model_llm', 'llama2:13b')
        n_results = int(data.get('n_results', 5))
        
        if not query_text or not collection_name:
            return jsonify({'error': 'Query text and collection name are required'}), 400
            
        # Perform the query
        result = query_with_rag(query_text, collection_name, n_results, model_embed, model_llm)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Detailed error: {str(e)}")
        print(traceback.format_exc())  # Print full traceback
        return jsonify({'error': f"Error during hybrid RAG process: {str(e)}"}), 500

@app.route('/rag/collections')
def rag_list_collections():
    """List available collections for RAG project"""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collections = client.list_collections()
        collection_names = [coll.name for coll in collections]
        
        return render_template('collections2.html', collections=collection_names)
        
    except Exception as e:
        flash(f'Error listing collections: {str(e)}', 'error')
        return redirect(url_for('rag_index'))

@app.route('/rag/select_collection/<collection_name>')
def rag_select_collection(collection_name):
    """Select a collection for querying in RAG project"""
    session['collection_name'] = collection_name
    flash(f'Selected collection: {collection_name}', 'success')
    return redirect(url_for('rag_query_page'))

# SQL Project Routes
@app.route('/sql')
def sql_index():
    """Home page for SQL project"""
    databases = get_available_databases()
    return render_template('sql_index.html', databases=databases)

@app.route('/sql/view_database', methods=['POST'])
def sql_view_database():
    """View database in SQL project"""
    db_name = request.form.get('database')
    sample_data, table_name = get_database_sample(db_name)
    databases = get_available_databases()
    return render_template('sql_index.html', 
                          databases=databases, 
                          selected_db=db_name, 
                          sample_data=sample_data, 
                          table_name=table_name)

@app.route('/sql/upload', methods=['POST'])
def sql_upload_file():
    """Handle file uploads for SQL project"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    db_name = request.form.get('new_db_name')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if not db_name:
        return jsonify({"error": "No database name provided"})
    
    # Save uploaded file temporarily
    filename = secure_filename(file.filename)
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, filename)
    file.save(file_path)
    
    # Upload to MySQL
    result = upload_excel_to_mysql(file_path, db_name)
    
    # Clean up temporary file
    os.remove(file_path)
    os.rmdir(temp_dir)
    
    return redirect(url_for('sql_index'))

@app.route('/sql/query', methods=['POST'])
def sql_query():
    """Handle SQL queries with REX analysis"""
    db_name = request.form.get('db_name')
    question = request.form.get('question')
    
    # Connect to the selected database
    db = SQLDatabase.from_uri(f"mysql+pymysql://root:root@localhost/{db_name}")
    
    # Create SQL query chain
    sql_query = create_sql_query_chain(llm, db)
    execute_query = QuerySQLDataBaseTool(db=db)
    
    try:
        # Get the context and raw SQL
        context = sql_query.invoke({'question': question})
        raw_sql = get_correct_sql_query.invoke({'context': context, 'question': question})
        
        # Sanitize the SQL query
        sanitized_sql = sanitize_sql(raw_sql)
        
        # Execute the sanitized query directly
        response = execute_query.invoke(sanitized_sql)
        
        # Parse the results for table display
        parsed_results = parse_sql_results(response)
        result_count = len(parsed_results) if parsed_results else 0
        
        # Apply REX analysis
        rex_analysis = analyze_results_with_rex(question, response, result_count)
        
        return jsonify({
            "result": response,
            "parsed_results": parsed_results,
            "result_count": result_count,
            "sql_query": sanitized_sql,
            "rex_analysis": rex_analysis
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)