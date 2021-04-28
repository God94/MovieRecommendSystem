package com.wildgoose.kafkaStream;

import org.apache.kafka.streams.KafkaStreams;
import org.apache.kafka.streams.StreamsBuilder;
import org.apache.kafka.streams.StreamsConfig;
import org.apache.kafka.streams.Topology;


import java.util.Properties;

public class Application {

    public static void main(String[] args) {

        String input = "abc";
        String output = "recommender";

        Properties properties = new Properties();
        properties.put(StreamsConfig.APPLICATION_ID_CONFIG, "logProcessor");
        properties.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG,"linux:9092");

        // 建立kafka拓扑
        StreamsBuilder builder = new StreamsBuilder();
        Topology topology = builder.build();
        topology.addSource("source", input)
                .addProcessor("process", () -> new LogProcessor(), "source")
                .addSink("sink", output, "process");

        KafkaStreams kafkaStreams = new KafkaStreams(topology, properties);

        kafkaStreams.start();
    }

}
