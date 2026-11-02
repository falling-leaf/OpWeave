/*
 * @Author: Anonymous anonymous@example.com
 * @Date: 2023-05-04 18:20:29
 * @LastEditors: Anonymous anonymous@example.com
 * @LastEditTime: 2023-06-30 11:41:56
 * @FilePath: /postgres-DB4AI/src/udf/spi_connection.h
 * @Description: 
 */
#pragma once
#ifndef _SPI_CONNECTION_H_
#define _SPI_CONNECTION_H_

#include <iostream>
#include <string>
#include <vector>


extern "C" {
#include "executor/spi.h"
}

//extern "C" {
//}



// internal SPI connection class for pg
class SPIConnector {
public:
    SPIConnector();
    ~SPIConnector();

    /**
     * @description: Create an SPI connection
     * @event: 
     * @return {*}
     */    
    bool Connect();

    /**
     * @description: Close the SPI connection
     * @event: 
     * @return {*}
     */    
    void DisConnect();

    /**
     * @description: Return the connection status
     * @event: 
     * @return {*}
     */    
    bool IsConnected();

    /**
     * @description: 
     * @event: 
     * @return {*}
     */    
    bool IsPrepared();

    /**
     * @description: Prepare the prepared statement
     * @event: 
     * @param {string&} query
     * @return {*}
     */    
    bool Prepare(const std::string& query, 
                 std::vector<Oid>& arg_types);


    bool PrepareExecute(std::vector<Datum>& values);
    bool Execute(const std::string& query);

    
private:
    /**
     * pg forbit parallel thread to do spi,
     * before call Bind and Execute
    */
    static inline std::mutex  conn_lock_;
    bool              is_connected_; // connection flag
    bool              is_prepared_;  // prepared flag
    SPIPlanPtr        plan_;         
};


class SPISqlWrapper{
public:
    SPISqlWrapper(SPIConnector& conn, 
                  const std::string& sql, 
                  const int parameters);

    ~SPISqlWrapper();

    /**
     * @description: 
     * @event: 
     * @param {int} index
     * @param {Oid} type
     * @param {Datum} value
     * @return {*}
     */    
    bool Bind(int index, 
              Oid type, 
              Datum value);
    
    /**
     * @description: 
     * @event: 
     * @return {*}
     */    
    bool Execute();

private:
    bool Prepare();

private:
    SPIConnector&             pg_conn_;
    std::string               sql_;             //sql statement
    std::vector<Oid>          param_types_;     
    std::vector<Datum>        param_values_;
};


#endif