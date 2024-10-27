Here's an updated README with Docker instructions to pass environment variables:

---

# Document Processing API

A FastAPI-based service for processing and analyzing PDF documents. This application uses embeddings to generate extended metadata, such as title, summaries, and tags, for uploaded documents, and integrates with backend services for further processing.

## Project Structure

-   **app**: Core application directory.
    -   **api/routes.py**: Defines API routes for document upload and health checks.
    -   **config.py**: Configuration settings for external dependencies.
    -   **main.py**: Initializes and runs the FastAPI application.
    -   **models/schemas.py**: Defines schemas for document metadata.
    -   **pipeline/pdf_pipeline.py**: Document processing pipeline, including text splitting and embedding-based analysis.
    -   **util/llm_provider.py**: Interface for connecting to LLM (large language model) providers.
-   **Dockerfile**: Configuration for containerizing the application.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/drop-table-team/PDFModule.git
    cd PDFModule
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Configure environment variables:

    - Create a `.env` file based on `.env.example` and set:
        - `OLLAMA_BASE_URL`: Base URL for LLM API.
        - `OLLAMA_MODEL`: Model identifier.
        - `BACKEND_BASE_URL`: Backend service URL for forwarding document data.

4. Run the application:
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 80
    ```

## API Endpoints

-   **GET /health**: Health check, verifies API status and LLM connection.
-   **POST /input**: Accepts PDF file uploads, processes them, and returns generated metadata.

## Usage

-   Upload a PDF to `/input` to receive document metadata such as title, summary, short summary, and tags.
-   Verify service and LLM connection status using `/health`.

## Docker

To build and run the application in a Docker container, follow these steps:

1. Build the Docker image:

    ```bash
    docker build -t document-processing-api .
    ```

2. Run the container, passing environment variables from your `.env` file:
    ```bash
    docker run --env-file .env -p 80:80 document-processing-api
    ```

The application will now be accessible at `http://localhost:80`.

---

This should set up the container with the necessary environment variables.
