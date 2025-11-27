<template>
  <div>
    <div class="wsi-retrieval">
      <h1>{{ $t('wsi-retrieval') }}</h1>
      <a @click="closeRetrieval()">
        <span class="fas fa-times-circle"></span>
      </a>
    </div>


    <div>
      <h5>{{ $t('similar-images') }}</h5>
      <button class="button is-small is-fullwidth" @click="searchSimilarImages">
        {{ $t('search-similar-image') }}
      </button>

      {{ data }}
    </div>
  </div>
</template>

<script>
import {Cytomine} from '@/api';

export default {
  data() {
    return {
      data: null,
    };
  },
  methods: {
    async searchSimilarImages() {
      this.data = (await Cytomine.instance.api.get(
        'wsi-cbir/retrieval',
        {
          params: {
            k: 10,
          }
        }
      )).data;
    }
  }
};
</script>