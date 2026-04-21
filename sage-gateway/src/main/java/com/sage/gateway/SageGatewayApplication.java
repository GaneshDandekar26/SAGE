package com.sage.gateway;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableAsync;

@SpringBootApplication
@EnableAsync
public class SageGatewayApplication {

    public static void main(String[] args) {
        SpringApplication.run(SageGatewayApplication.class, args);
    }

}
