/*
 * @Author: Anonymous anonymous@example.com
 * @Date: 2023-04-24 18:15:44
 * @LastEditors: Anonymous anonymous@example.com
 * @LastEditTime: 2023-05-08 23:45:54
 * @FilePath: /postgres-DB4AI/src/udf/connection.h
 * @Description: 
 */
#pragma once
#ifndef _CONNECTION_H_
#define _CONNECTION_H_

#include <iostream>
#include <string>
#include <vector>
#include "libpq-fe.h"

struct LastError {
    ConnStatusType error_code;
    std::string   error_message;
};

// pg external connection class
class PostgresConnector {
public:
    PostgresConnector();

    ~PostgresConnector();

    /**
     * @description: 
     * @event: 
     * @param {string&} connstr
     * @return {*}
     */    
    bool  ConnectToDB(const std::string& connstr);

    /**
     * @description: 
     * @event: 
     * @param {string&} host
     * @param {string&} user
     * @param {string&} password
     * @param {string&} database
     * @param {string&} port
     * @return {*}
     */    
    bool  ConnectToDB(const std::string& host, const std::string& user, const std::string& password, 
                     const std::string& database, const std::string& port="5432");

    /**
     * @description: 
     * @event: 
     * @return {*}
     */    
    bool  Begin();

    /**
     * @description: 
     * @event: 
     * @return {*}
     */    
    bool  Commit() ;

    /**
     * @description: 
     * @event: 
     * @return {*}
     */    
    bool  Rollback();

    /**
     * @description: 
     * @event: 
     * @param {string&} charset
     * @return {*}
     */    
    bool  SetCharset(const std::string& charset);

    /**
     * @description: 
     * @event: 
     * @return {*}
     */    
    void  DisConnect();

    /**
     * @description: 
     * @event: 
     * @return {*}
     */    
    inline bool IsConnected();

    /**
     * @description: 
     * @event: 
     * @return {*}
     */    
    PGconn* GetConnection() const;

    /**
     * @description: 
     * @event: 
     * @return {*}
     */    
    LastError GetLastError() const;

    
    /**
     * @description: Execute a regular SQL statement
     * @event: 
     * @param {string&} sql
     * @return {*}
     */    
    bool  Execute(const std::string& sql, int& tup_num);

    /**
     * @description:  Prepare a statement
     * @event: 
     * @param {string&} stmtName
     * @param {string&} sql
     * @param {int} nParams
     * @param {Oid*} paramTypes
     * @return {*}
     */    
    PGresult*  Prepare(const std::string& stmt_name, const std::string& sql, 
                  int nparams, const Oid* param_types);

    /**
     * @description: Execute the prepared statement
     * @event: 
     * @param {string&} stmtName
     * @param {int} nParams
     * @param {char* const*} paramValues
     * @param {int*} paramLengths
     * @param {int*} paramFormats
     * @param {int} resultFormat
     * @param {int} tup_num 
     * @return {*}
     */    
    bool  PrepareExecute(const std::string& stmt_name, int nparams, const char* const* param_values, 
                         const int* param_lengths, const int* param_formats, int result_format, int& tup_num);
private:
    PGconn*              conn_;       //pg connection handle
    //std::string          sql_string_; //sql statement
    LastError            last_error_;  //last error that occurred
};

class SqlWrapper {

public:
    SqlWrapper(PostgresConnector& conn);
    SqlWrapper(PostgresConnector& conn, const std::string& sql);
    
    ~SqlWrapper();


    /**
     * @description: 
     * @event: 
     * @param {int} index
     * @param {char*} value
     * @param {int} length
     * @param {int} format
     * @return {*}
     */       
    bool  Bind(int index, const char* value, int length, int format);

    /**
     * @description: 
     * @event: 
     * @param {int} index
     * @param {string&} value
     * @return {*}
     */    
    bool  Bind(int index, const std::string& value);

    /**
     * @description: 
     * @event: 
     * @param {int} index
     * @param {int} value
     * @return {*}
     */    
    bool  Bind(int index, int value);

    /**
     * @description: 
     * @event: 
     * @param {int} index
     * @param {double} value
     * @return {*}
     */    
    bool  Bind(int index, double value);

    /**
     * @description: 
     * @event: 
     * @param {int} tup_num number of modified rows
     * @return {*}
     */    
    bool  Execute(int& tup_num);
private:
    /**
     * @description: 
     * @event: 
     * @return {*}
     */    
    void  Prepare();

private:
    PostgresConnector&              pg_conn_;       //pg connection class 
    std::string                     sql_;           //sql statement
    std::string                     stmt_name_;
    std::vector<const char*>        param_values_;  
    std::vector<int>                param_lengths_;
    std::vector<int>                param_formats_;
    std::vector<Oid>                param_types_;
};

#endif