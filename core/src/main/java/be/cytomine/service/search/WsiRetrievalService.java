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
import be.cytomine.domain.image.AbstractImage;
import be.cytomine.dto.image.CropParameter;
import be.cytomine.dto.search.SearchResponse;
import be.cytomine.service.middleware.ImageServerService;

@Slf4j
@RequiredArgsConstructor
@Service
public class WsiRetrievalService {

    public static final String CBIR_API_BASE_PATH = "http://wsi-cbir:6001";

    private final RestTemplate restTemplate;

    @Value("${application.cbirURL}")
    private String cbirUrl;

    public String getInternalCbirURL() {
        return CBIR_API_BASE_PATH;
    }

    public ResponseEntity<String> retrieveSimilarImages(Long k, String query, String datasets, String staining, String organ, String species, String diagnosis) {
        String url = UriComponentsBuilder
            .fromHttpUrl(getInternalCbirURL())
            .path("/api/retrieval")
            .queryParam("query", query)
            .queryParam("datasets", datasets)
            .queryParam("staining", staining)
            .queryParam("organ", organ)
            .queryParam("species", species)
            .queryParam("diagnosis", diagnosis)
            .queryParam("k", k)
            .toUriString();
        log.debug(url);
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(null, headers);
        return restTemplate.exchange(url, HttpMethod.POST, entity, String.class);
    }

    public ResponseEntity<String> indexImage(AbstractImage image) {
        URI url = UriComponentsBuilder
            .fromHttpUrl(getInternalCbirURL())
            .path("/api/indexing")
            .queryParam("image_id", image.getId())
            .queryParam("path", image.getPath())
            .queryParam("filename", image.getOriginalFilename())
            .build()
            .toUri();

        log.debug("Create index for image {}", image.getId());

        return restTemplate.exchange(url, HttpMethod.POST, null, String.class);
    }

    public ResponseEntity<String> removeImage(AbstractImage image) {
        URI url = UriComponentsBuilder
            .fromHttpUrl(getInternalCbirURL())
            .path("/api/rm")
            .queryParam("image_id", image.getId())
            .queryParam("path", image.getPath())
            .queryParam("filename", image.getOriginalFilename())
            .build()
            .toUri();

        log.debug("Remove index for image {}", image.getId());

        return restTemplate.exchange(url, HttpMethod.POST, null, String.class);
    }

    public ResponseEntity<SearchResponse> getJob(String jobId) {
        URI url = UriComponentsBuilder
            .fromHttpUrl(getInternalCbirURL())
            .path("/api/jobs/{job_id}")
            .queryParam("job_id", jobId)
            .build()
            .toUri();

        ResponseEntity<SearchResponse> response = restTemplate.exchange(url, HttpMethod.GET, null, SearchResponse.class);
        log.debug("Receiving response {}", response);

        SearchResponse searchResponse = response.getBody();
        return ResponseEntity.ok(searchResponse);
        
    }
}