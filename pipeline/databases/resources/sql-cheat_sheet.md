# SQL Command Cheat Sheet

## Table of Contents
1. [Database Operations](#database-operations)
2. [Table Operations](#table-operations)
3. [Data Manipulation](#data-manipulation)
4. [Querying Data](#querying-data)
5. [Joins](#joins)
6. [Aggregate Functions](#aggregate-functions)
7. [Subqueries](#subqueries)
8. [Indexes](#indexes)
9. [Views](#views)
10. [Transactions](#transactions)
11. [User and Privileges](#user-and-privileges)

---

## Database Operations
- **Create Database**:
  ```sql
  CREATE DATABASE database_name;
  ```

- **Drop Database**:
  ```sql
  DROP DATABASE database_name;
  ```

## Table Operations
- **Create Table**:
  ```sql
  CREATE TABLE table_name (
    column1 datatype,
    column2 datatype,
    ...
  );
  ```

- **Drop Table**:
  ```sql
  DROP TABLE table_name;
  ```

- **Alter Table**:
  - **Add Column:**
    ```sql
    ALTER TABLE table_name ADD column_name dataype;
    ```

  - **Drop Column:**
    ```sql
    ALTER TABLE table_name DROP COLUMN column_name;
    ```

- **Modify Column**:
  ```sql
  ALTER TABLE table_name MODIFY COLUMN column_name datatype;
  ```

## Data Manipulation

- **Insert Data:**
  ```sql
  INSERT INTO table_name (column1, column2, ...) VALUES (value1, value2, ...);
  ```

- **Update Data:**
  ```sql
  UPDATE table_name SET column1 = value1, column2 = value2, ... WHERE condition;
  ```

- **Delete Data:**
  ```sql
  DELETE FROM table_name WHERE condition;
  ```

## Querying Data

- **Select All Data:**
  ```sql
  SELECT * FROM table_name;
  ```

- **Select Data:**
  ```sql
  SELECT column1, column2, ... FROM table_name;
  ```

- **Select Disctint Data:**
  ```sql
  SELECT DISTINCT column1 FROM table_name;
  ```

- **Where Clause:**
  ```sql
  SELECT column1, column2, ... FROM table_name WHERE condition;
  ```

- **Order By Clause:**
  ```sql
  SELECT column1, column2, ... FROM table_name ORDER BY column1 [ASC|DESC];
  ```

- **Limit Clause:**
  ```sql
  SELECT column1, column2, ... FROM table_name LIMIT number;
  ```

## Joins

- **Inner Join:**
  ```sql
  SELECT columns FROM table1 INNER JOIN table2 ON table1.column = table2.column;
  ```

- **Left Join:**
  ```sql
  SELECT columns FROM table1 LEFT JOIN table2 ON table1.column = table2.column;
  ```

- **Right Join:**
  ```sql
  SELECT columns FROM table1 RIGHT JOIN table2 ON table1.column = table2.column;
  ```

- **Full Join:**
  ```sql
  SELECT columns FROM table1 FULL OUTER JOIN table2 ON table1.column = table2.column;
  ```

## Aggregate Functions

- **Count:**
  ```sql
  SELECT COUNT(column) FROM table_name;
  ```

- **Sum:**
  ```sql
  SELECT SUM(column) FROM table_name;
  ```

- **Average:**
  ```sql
  SELECT AVG(column) FROM table_name;
  ```

- **Max:**
  ```sql
  SELECT MAX(column) FROM table_name;
  ```

- **Min:**
  ```sql
  SELECT MIN(column) FROM table_name;
  ```

## Subqueries

- **Subquerry in Select:**
  ```sql
  SELECT column, (SELECT column FROM table_name WHERE condition) AS alias FROM table_name;
  ```

- **Subquerry in WHERE:**
  ```sql
  SELECT column FROM table_name WHERE column = (SELECT column FROM table_name WHERE condition);
  ```

## Indexes

- **Create Index:**
  ```sql
  CREATE INDEX index_name ON table_name (column1, column2, ...);
  ```

- **Drop Index:**
  ```sql
  DROP INDEX index_name ON table_name;
  ```

## Views

- **Create View:**
  ```sql
  CREATE VIEW view_name AS SELECT column1, column2, ... FROM table_name WHERE condition;
  ```

- **Drop View:**
  ```sql
  DROP VIEW view_name;
  ```

## Transactions

- **Begin Transaction:**
  ```sql
  BEGIN;
  ```

- **Comit Transaction:**
  ```sql
  COMMIT;
  ```

- **Rollback Transaction:**
  ```sql
  ROLLBACK;
  ```

## User and Priviledges

- **Create User:**
  ```sql
  CREATE USER 'username'@'host' IDENTIFIED BY 'password';
  ```

- **Grant Privileges:**
  ```sql
  CREATE USER 'username'@'host' IDENTIFIED BY 'password';
  ```

- **Revoke Privileges:**
  ```sql
  CREATE USER 'username'@'host' IDENTIFIED BY 'password';
  ```

- **Drop User:**
  ```sql
  DROP USER 'username'@'host';
  ```