# llm-zoomcamp

Run elaasticsearch with docker:

```
docker run -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```

Note:

-e "ES_JAVA_OPTS=-Xms512m -Xmx512m" - you may need this environment variable if elasticsearch is not starting, but it's not nessesary
