package com.sage.gateway.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.ClassPathResource;
import org.springframework.data.redis.core.script.RedisScript;

@Configuration
public class RateLimitConfig {

    @Bean
    public RedisScript<Long> rateLimitScript(){
        ClassPathResource scriptSource = new ClassPathResource("scripts/rate-limit.lua");

        return RedisScript.of(scriptSource, Long.class);
    }

}
