package be.cytomine.service.search;

import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import be.cytomine.config.properties.ApplicationProperties;
import be.cytomine.domain.ontology.AnnotationDomain;
import be.cytomine.dto.image.CropParameter;
import be.cytomine.dto.search.SearchResponse;
import be.cytomine.service.middleware.ImageServerService;

@Slf4j
@RequiredArgsConstructor
@Service
public class WsiRetrievalService {

    public static final String CBIR_API_BASE_PATH = "/wsi-cbir";

    private final RestTemplate restTemplate;

    @Value("${application.cbirURL}")
    private String cbirUrl;

    public String getInternalCbirURL() {
        return cbirUrl + CBIR_API_BASE_PATH;
    }

    public ResponseEntity<SearchResponse> retrieveSimilarImages(Long k_best) {
        String url = UriComponentsBuilder
            .fromHttpUrl(getInternalCbirURL())
            .path("/retrieval")
            .queryParam("query", "/queries/IMAGE_AAeZemiStB")
            .queryParam("k_best", k_best + 1)
            .toUriString();

        // Create an empty HttpEntity for POST requests with no body
        HttpEntity<Void> entity = new HttpEntity<>(null);

        ResponseEntity<SearchResponse> response = this.restTemplate.exchange(
            url,
            HttpMethod.POST,
            entity,                     // required HttpEntity
            SearchResponse.class        // response type
        );

        log.debug("Receiving response {}", response);

        SearchResponse searchResponse = response.getBody();
        if (searchResponse == null) {
            return response;
        }

        return ResponseEntity.ok(searchResponse);
    }
}