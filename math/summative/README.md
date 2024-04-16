# linear-regression-summative

To run the notebook [linnear_regression_summative_[Kingsley Budu Boafo].ipynb], first install the requirements by using
- pip install -r requirements.txt

### Fast API Usage

1. Run the FastAPI server:

2. Visit `http://127.0.0.1:8000/docs` in your web browser to access the interactive API documentation (Swagger UI).

3. Test the API endpoints using the provided documentation.

## API Endpoints

- **GET /:** 
    - Description: A simple root endpoint to check the connection.
    - Example Response: "Successfully Connected"

- **POST /predict:**
    - Description: Endpoint to predict TV sales based on the input value.
    - Request Body: JSON object with a single field `tv` representing the TV sales value.
    - Example Request:
        ```json
        {
            "tv": 150
        }
        ```
    - Example Response: "Predicted Tv sales : [predicted_value]"

## Built With

- [FastAPI](https://fastapi.tiangolo.com/) - Web framework for building APIs with Python.
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation and settings management using Python type annotations.
- [uvicorn](https://www.uvicorn.org/) - ASGI server for running Python web applications.

## Authors

- [Kingsley Budu](https://github.com/kayc0des) - Initial work

To access the documentation, use;
- '/docs' end point